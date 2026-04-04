/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#include "ray_tracer.h"
#include "optix_ray.h"
#include <unordered_map>   // ← pour vmap dans le constructeur


#include <iostream>
#include <fstream>
#include <filesystem>
#include <limits>
#include <chrono>

#include <spdlog/spdlog.h>

#include <optix_function_table.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std::chrono;

// ─── Macro helpers ───────────────────────────────────────────────────────────

#define OPTIX_CHECK(call)                                                        \
    do {                                                                         \
        OptixResult _res = (call);                                               \
        if (_res != OPTIX_SUCCESS) {                                             \
            spdlog::error("OptiX error: {} at {}:{}",                           \
                          optixGetErrorString(_res), __FILE__, __LINE__);        \
        }                                                                        \
    } while (0)

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            spdlog::error("CUDA error: {} at {}:{}",                            \
                          cudaGetErrorString(_err), __FILE__, __LINE__);         \
        }                                                                        \
    } while (0)

// ─── OptiX log callback ──────────────────────────────────────────────────────

static void optixLogger(unsigned int level, const char* tag,
                         const char* message, void*)
{
    if      (level <= 2) spdlog::error("[OptiX][{}][{}] {}", level, tag, message);
    else if (level == 3) spdlog::warn ("[OptiX][{}][{}] {}", level, tag, message);
    else                 spdlog::trace("[OptiX][{}][{}] {}", level, tag, message);
}

// ─── Constructeur ────────────────────────────────────────────────────────────

RayTracer::RayTracer(SurfaceMesh& mesh) : mesh(mesh)
{
    spdlog::info("RayTracer init: {} faces", mesh.number_of_faces());

    // ── 1. Init CUDA ─────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(nullptr));   // force l'init du contexte CUDA

    // ── 2. Init OptiX ────────────────────────────────────────────────────────
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions ctxOpts{};
    ctxOpts.logCallbackFunction = optixLogger;
    ctxOpts.logCallbackLevel    = 4;
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &ctxOpts, &context));

    // ── 3. Build GAS ─────────────────────────────────────────────────────────
    // Extraction vertices + indices depuis le CGAL SurfaceMesh
    std::vector<float3>       h_vertices;
    std::vector<unsigned int> h_indices;

    // Map CGAL vertex_descriptor → index GPU
    std::unordered_map<SurfaceMesh::Vertex_index, unsigned int> vmap;
    vmap.reserve(mesh.number_of_vertices());

    for (auto v : mesh.vertices()) {
        const Point& p = mesh.point(v);
        vmap[v] = static_cast<unsigned int>(h_vertices.size());
        h_vertices.push_back(make_float3(
            static_cast<float>(p.x()),
            static_cast<float>(p.y()),
            static_cast<float>(p.z())));
    }

    for (auto f : mesh.faces()) {
        for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh))
            h_indices.push_back(vmap.at(v));
    }

    const size_t vertBytes = h_vertices.size() * sizeof(float3);
    const size_t idxBytes  = h_indices.size()  * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices),
                          h_vertices.data(), vertBytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), idxBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices),
                          h_indices.data(), idxBytes, cudaMemcpyHostToDevice));

    // Descriptor de géométrie triangulée
    OptixBuildInput triangleInput{};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangleInput.triangleArray.numVertices         = static_cast<unsigned int>(h_vertices.size());
    triangleInput.triangleArray.vertexBuffers       = &d_vertices;

    triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes  = 3 * sizeof(unsigned int);
    triangleInput.triangleArray.numIndexTriplets    = static_cast<unsigned int>(mesh.number_of_faces());
    triangleInput.triangleArray.indexBuffer         = d_indices;

    const unsigned int geomFlags = OPTIX_GEOMETRY_FLAG_NONE;
    triangleInput.triangleArray.flags               = &geomFlags;
    triangleInput.triangleArray.numSbtRecords       = 1;

    OptixAccelBuildOptions accelOpts{};
    accelOpts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                         | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context, &accelOpts, &triangleInput, 1, &gasBufferSizes));

    CUdeviceptr d_temp = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp),
                          gasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer),
                          gasBufferSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context, nullptr,
        &accelOpts, &triangleInput, 1,
        d_temp,            gasBufferSizes.tempSizeInBytes,
        d_gas_output_buffer, gasBufferSizes.outputSizeInBytes,
        &gas_handle, nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));

    // ── 4. Compile module PTX ─────────────────────────────────────────────────
    // Le PTX est embarqué via CMake (target_embed_ptx ou xxd)
    // Assurez-vous que optix_ray.ptx est accessible au runtime
    std::string ptxPath = "RayTracer.optixir";
    std::ifstream ptxFile(ptxPath, std::ios::binary);
    if (!ptxFile)
        spdlog::error("Cannot open OptiX IR file: {}", ptxPath);

    std::string ptxSource((std::istreambuf_iterator<char>(ptxFile)),
                            std::istreambuf_iterator<char>());

    OptixModuleCompileOptions moduleOpts{};
    moduleOpts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipelineCompileOpts{};
    pipelineCompileOpts.usesMotionBlur                   = 0;
    pipelineCompileOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOpts.numPayloadValues                 = 2;   // hitFlag + hitFaceId
    pipelineCompileOpts.numAttributeValues               = 2;
    pipelineCompileOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOpts.pipelineLaunchParamsVariableName = "params";

    OPTIX_CHECK(optixModuleCreate(
        context,
        &moduleOpts,
        &pipelineCompileOpts,
        ptxSource.c_str(), ptxSource.size(),
        nullptr, nullptr,
        &module));

    // ── 5. Program groups ────────────────────────────────────────────────────
    OptixProgramGroupOptions pgOpts{};

    // RayGen
    OptixProgramGroupDesc rgDesc{};
    rgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module            = module;
    rgDesc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(
        context, &rgDesc, 1, &pgOpts, nullptr, nullptr, &raygen_prog_group));

    // Miss
    OptixProgramGroupDesc msDesc{};
    msDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module            = module;
    msDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(
        context, &msDesc, 1, &pgOpts, nullptr, nullptr, &miss_prog_group));

    // Hitgroup
    OptixProgramGroupDesc chDesc{};
    chDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    chDesc.hitgroup.moduleCH            = module;
    chDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK(optixProgramGroupCreate(
        context, &chDesc, 1, &pgOpts, nullptr, nullptr, &hitgroup_prog_group));

    // ── 6. Pipeline ──────────────────────────────────────────────────────────
    OptixProgramGroup groups[] = {
        raygen_prog_group, miss_prog_group, hitgroup_prog_group };

    OptixPipelineLinkOptions linkOpts{};
    linkOpts.maxTraceDepth = 1;

    OPTIX_CHECK(optixPipelineCreate(
        context, &pipelineCompileOpts, &linkOpts,
        groups, 3, nullptr, nullptr, &pipeline));

    // ── 7. SBT ───────────────────────────────────────────────────────────────
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    auto uploadRecord = [&](OptixProgramGroup pg, CUdeviceptr& devPtr) {
        EmptyRecord rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devPtr), sizeof(EmptyRecord)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(devPtr), &rec,
                              sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    };

    uploadRecord(raygen_prog_group,   raygen_record);
    uploadRecord(miss_prog_group,     miss_record);
    uploadRecord(hitgroup_prog_group, hitgroup_record);

    sbt = {};
    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = sizeof(EmptyRecord);
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.hitgroupRecordCount         = 1;

    spdlog::info("RayTracer ready ({} vertices, {} faces)",
                 h_vertices.size(), mesh.number_of_faces());
}


// ─── computeRaysAndHits ──────────────────────────────────────────────────────

std::vector<float> RayTracer::computeRaysAndHits(
    const Point&                   origin,
    OptixDeviceContext              optixContext,
    OptixTraversableHandle          gasHandle,
    OptixPipeline                   pipelineArg,
    const OptixShaderBindingTable&  sbtArg) const
{
    // ── Constantes pour tmax ─────────────────────────────────────────────────
    constexpr float TMAX_OFFSET_REL = 0.01f;   // 1% de la distance
    constexpr float TMAX_OFFSET_MIN = 1e-3f;   // 1mm minimum
    constexpr float TMAX_OFFSET_MAX = 0.5f;    // 50cm maximum

    // ── Build ray list ───────────────────────────────────────────────────────
    std::vector<OptixRay> rays;
    std::vector<float>    distances;
    rays.reserve(mesh.number_of_faces());
    distances.reserve(mesh.number_of_faces());

    const float3 orig = make_float3(
        static_cast<float>(origin.x()),
        static_cast<float>(origin.y()),
        static_cast<float>(origin.z()));

    unsigned int faceIdx = 0u;
    for (auto f : mesh.faces()) {

        // Centroïde du triangle
        std::vector<Point> tri;
        tri.reserve(3);
        for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh))
            tri.push_back(mesh.point(v));

        if (tri.size() != 3) {
            // Face dégénérée : index conservé pour cohérence
            distances.push_back(std::numeric_limits<float>::max());
            rays.push_back({ orig, make_float3(0.f, 0.f, 1.f), 0.f, faceIdx });
            ++faceIdx;
            continue;
        }

        const Point centroid = CGAL::centroid(tri[0], tri[1], tri[2]);

        float dx  = static_cast<float>(centroid.x() - origin.x());
        float dy  = static_cast<float>(centroid.y() - origin.y());
        float dz  = static_cast<float>(centroid.z() - origin.z());
        float len = sqrtf(dx*dx + dy*dy + dz*dz);

        if (len < 1e-6f) {
            distances.push_back(0.0f);
            rays.push_back({ orig, make_float3(0.f, 0.f, 1.f), 0.f, faceIdx });
            ++faceIdx;
            continue;
        }

        const float3 dir = make_float3(dx / len, dy / len, dz / len);

        // tmax adaptatif : s'arrêter avant d'atteindre la face cible
        const float offset = std::min(
            std::max(TMAX_OFFSET_REL * len, TMAX_OFFSET_MIN),
            TMAX_OFFSET_MAX);
        const float tmax = len - offset;

        distances.push_back(len);
        rays.push_back({ orig, dir, tmax, faceIdx });
        ++faceIdx;
    }

    const size_t numRays = rays.size();
    spdlog::debug("Building {} rays", numRays);

    // ── Upload rays vers GPU ─────────────────────────────────────────────────
    OptixRay* d_rays = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rays),
                          numRays * sizeof(OptixRay)));
    CUDA_CHECK(cudaMemcpy(d_rays, rays.data(),
                          numRays * sizeof(OptixRay),
                          cudaMemcpyHostToDevice));

    uint8_t* d_hits = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hits),
                          numRays * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_hits, 0, numRays * sizeof(uint8_t)));

    // ── Paramètres de lancement ──────────────────────────────────────────────
    RayGenLaunchParams h_params;
    h_params.rays      = d_rays;
    h_params.hits      = d_hits;
    h_params.numRays   = static_cast<unsigned int>(numRays);
    h_params.gasHandle = gasHandle;

    RayGenLaunchParams* d_params = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params),
                          sizeof(RayGenLaunchParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &h_params,
                          sizeof(RayGenLaunchParams),
                          cudaMemcpyHostToDevice));

    // ── Lancement OptiX ──────────────────────────────────────────────────────
    OPTIX_CHECK(optixLaunch(
        pipelineArg,
        0,                                          // stream CUDA
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(RayGenLaunchParams),
        &sbtArg,
        static_cast<unsigned int>(numRays),         // width
        1u,                                         // height
        1u                                          // depth
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Récupération des résultats ───────────────────────────────────────────
    std::vector<uint8_t> h_hits(numRays);
    CUDA_CHECK(cudaMemcpy(h_hits.data(), d_hits,
                          numRays * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));

    // ── Libération mémoire GPU temporaire ────────────────────────────────────
    CUDA_CHECK(cudaFree(d_rays));
    CUDA_CHECK(cudaFree(d_hits));
    CUDA_CHECK(cudaFree(d_params));

    // ── Construction du résultat ─────────────────────────────────────────────
    // Convention de retour :
    //   distance réelle  → face visible
    //  -1.0f             → face occultée
    std::vector<float> result;
    result.reserve(numRays);

    for (size_t i = 0; i < numRays; ++i) {
        if (h_hits[i] != 0u)
            result.push_back(-1.0f);        // occultée
        else
            result.push_back(distances[i]); // visible, distance en mètres
    }

    spdlog::debug("{} visible faces, {} occluded",
        std::count(h_hits.begin(), h_hits.end(), 0u),
        std::count(h_hits.begin(), h_hits.end(), 1u));

    return result;
}


// ─── traceRay (single point) ─────────────────────────────────────────────────

std::vector<float> RayTracer::traceRay(const Point& p) const
{
    auto start = high_resolution_clock::now();
    spdlog::debug("traceRay called for single point");

    auto distances = computeRaysAndHits(p, context, gas_handle, pipeline, sbt);

    auto ms = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    spdlog::debug("{} rays computed in {} ms", distances.size(), ms);
    return distances;
}

// ─── traceRay (multiple points) ──────────────────────────────────────────────

std::vector<std::vector<float>> RayTracer::traceRay(const std::vector<Point>& points) const
{
    auto start = high_resolution_clock::now();
    spdlog::debug("traceRay called for {} points", points.size());

    std::vector<std::vector<float>> result;
    result.reserve(points.size());
    for (const auto& p : points)
        result.push_back(computeRaysAndHits(p, context, gas_handle, pipeline, sbt));

    auto ms = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    if (!result.empty())
        spdlog::debug("{} rays computed in {} ms",
                      result[0].size() * points.size(), ms);
    return result;
}

// ─── cleanup ─────────────────────────────────────────────────────────────────

void RayTracer::cleanup()
{
    if (raygen_record)    { cudaFree(reinterpret_cast<void*>(raygen_record));   raygen_record   = 0; }
    if (miss_record)      { cudaFree(reinterpret_cast<void*>(miss_record));     miss_record     = 0; }
    if (hitgroup_record)  { cudaFree(reinterpret_cast<void*>(hitgroup_record)); hitgroup_record = 0; }
    if (d_vertices)       { cudaFree(reinterpret_cast<void*>(d_vertices));      d_vertices      = 0; }
    if (d_indices)        { cudaFree(reinterpret_cast<void*>(d_indices));       d_indices       = 0; }
    if (d_gas_output_buffer) {
        cudaFree(reinterpret_cast<void*>(d_gas_output_buffer));
        d_gas_output_buffer = 0;
    }

    if (pipeline)           { optixPipelineDestroy(pipeline);           pipeline           = nullptr; }
    if (raygen_prog_group)  { optixProgramGroupDestroy(raygen_prog_group);  raygen_prog_group  = nullptr; }
    if (miss_prog_group)    { optixProgramGroupDestroy(miss_prog_group);    miss_prog_group    = nullptr; }
    if (hitgroup_prog_group){ optixProgramGroupDestroy(hitgroup_prog_group);hitgroup_prog_group= nullptr; }
    if (module)             { optixModuleDestroy(module);               module             = nullptr; }
    if (context)            { optixDeviceContextDestroy(context);       context            = nullptr; }

    spdlog::debug("RayTracer resources cleaned up");
}
