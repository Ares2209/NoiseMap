/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#include "ray_tracer.h"
#include "optix_ray.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <unordered_map>

#include <spdlog/spdlog.h>

#include <optix_function_table.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

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
    num_faces = static_cast<unsigned int>(mesh.number_of_faces());
    spdlog::info("RayTracer init: {} faces", num_faces);

    // ── 1. Init CUDA + OptiX ─────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(nullptr));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions ctxOpts{};
    ctxOpts.logCallbackFunction = optixLogger;
    ctxOpts.logCallbackLevel    = 4;
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &ctxOpts, &context));

    // ── 2. Extraction des vertices/indices + précalcul des centroïdes ────────
    std::vector<float3>       h_vertices;
    std::vector<unsigned int> h_indices;

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
        unsigned int n = 0;
        for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh)) {
            if (n < 3)
                h_indices.push_back(vmap.at(v));
            ++n;
        }
        if (n != 3) {
            spdlog::error("RayTracer: face {} has {} vertices (expected 3 after triangulation)", f.idx(), n);
        }
    }

    // Précalcul des centroïdes de chaque face (une seule fois)
    h_centroids.reserve(num_faces);
    unsigned int non_tri_faces = 0;
    for (auto f : mesh.faces()) {
        Point pts[3];
        unsigned int n = 0;
        for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh)) {
            if (n < 3) pts[n] = mesh.point(v);
            ++n;
        }

        if (n == 3) {
            const Point c = CGAL::centroid(pts[0], pts[1], pts[2]);
            h_centroids.push_back(make_float3(
                static_cast<float>(c.x()),
                static_cast<float>(c.y()),
                static_cast<float>(c.z())));
        } else {
            // OptiX ici ne traite correctement que les triangles.
            // Les faces non triangulaires sont marquées invalides.
            ++non_tri_faces;
            h_centroids.push_back(make_float3(
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()));
        }
    }

    // ── 3. Upload vers GPU ───────────────────────────────────────────────────
    const size_t vertBytes     = h_vertices.size()  * sizeof(float3);
    const size_t idxBytes      = h_indices.size()   * sizeof(unsigned int);
    const size_t centroidBytes = h_centroids.size() * sizeof(float3);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices),
                          h_vertices.data(), vertBytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), idxBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices),
                          h_indices.data(), idxBytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centroids), centroidBytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_centroids),
                          h_centroids.data(), centroidBytes, cudaMemcpyHostToDevice));

    if (non_tri_faces > 0) {
        spdlog::warn("RayTracer: {} non-triangular faces will be ignored by centroid tracing",
                     non_tri_faces);
    }

    // Buffers persistants pour le lancement (réutilisés à chaque appel)
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hits_buf),
                          num_faces * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params_buf),
                          sizeof(RayGenLaunchParams)));

    // ── 4. Build GAS ─────────────────────────────────────────────────────────
    OptixBuildInput triangleInput{};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangleInput.triangleArray.numVertices         = static_cast<unsigned int>(h_vertices.size());
    triangleInput.triangleArray.vertexBuffers       = &d_vertices;

    triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes  = 3 * sizeof(unsigned int);
    triangleInput.triangleArray.numIndexTriplets    = num_faces;
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
        d_temp,              gasBufferSizes.tempSizeInBytes,
        d_gas_output_buffer, gasBufferSizes.outputSizeInBytes,
        &gas_handle, nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));

    // ── 5. Compile module OptiX IR ───────────────────────────────────────────
    std::ifstream irFile("RayTracer.optixir", std::ios::binary);
    if (!irFile)
        spdlog::error("Cannot open OptiX IR file: RayTracer.optixir");

    std::string irSource((std::istreambuf_iterator<char>(irFile)),
                          std::istreambuf_iterator<char>());

    OptixModuleCompileOptions moduleOpts{};
    moduleOpts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipelineCompileOpts{};
    pipelineCompileOpts.usesMotionBlur                   = 0;
    pipelineCompileOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOpts.numPayloadValues                 = 2;
    pipelineCompileOpts.numAttributeValues               = 2;
    pipelineCompileOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOpts.pipelineLaunchParamsVariableName = "params";

    OPTIX_CHECK(optixModuleCreate(
        context, &moduleOpts, &pipelineCompileOpts,
        irSource.c_str(), irSource.size(),
        nullptr, nullptr, &module));

    // ── 6. Program groups ────────────────────────────────────────────────────
    OptixProgramGroupOptions pgOpts{};

    OptixProgramGroupDesc rgDesc{};
    rgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module            = module;
    rgDesc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(
        context, &rgDesc, 1, &pgOpts, nullptr, nullptr, &raygen_prog_group));

    OptixProgramGroupDesc msDesc{};
    msDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module            = module;
    msDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(
        context, &msDesc, 1, &pgOpts, nullptr, nullptr, &miss_prog_group));

    OptixProgramGroupDesc chDesc{};
    chDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    chDesc.hitgroup.moduleCH            = module;
    chDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK(optixProgramGroupCreate(
        context, &chDesc, 1, &pgOpts, nullptr, nullptr, &hitgroup_prog_group));

    // ── 7. Pipeline ──────────────────────────────────────────────────────────
    OptixProgramGroup groups[] = {
        raygen_prog_group, miss_prog_group, hitgroup_prog_group };

    OptixPipelineLinkOptions linkOpts{};
    linkOpts.maxTraceDepth = 1;

    OPTIX_CHECK(optixPipelineCreate(
        context, &pipelineCompileOpts, &linkOpts,
        groups, 3, nullptr, nullptr, &pipeline));

    // ── 8. SBT ───────────────────────────────────────────────────────────────
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

    spdlog::info("RayTracer ready ({} vertices, {} faces, centroids precomputed)",
                 h_vertices.size(), num_faces);
}

// ─── computeRaysAndHits ──────────────────────────────────────────────────────

std::vector<float> RayTracer::computeRaysAndHits(const Point& origin) const
{
    const float3 orig = make_float3(
        static_cast<float>(origin.x()),
        static_cast<float>(origin.y()),
        static_cast<float>(origin.z()));

    // Remplir les paramètres de lancement (seule l'origin change entre appels)
    RayGenLaunchParams h_params;
    h_params.origin        = orig;
    h_params.centroids     = reinterpret_cast<float3*>(d_centroids);
    h_params.hits          = d_hits_buf;
    h_params.numRays       = num_faces;
    h_params.gasHandle     = gas_handle;
    h_params.tmaxOffsetRel = 0.01f;
    h_params.tmaxOffsetMin = 1e-3f;
    h_params.tmaxOffsetMax = 0.5f;

    CUDA_CHECK(cudaMemset(d_hits_buf, 0, num_faces * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_params_buf, &h_params,
                          sizeof(RayGenLaunchParams),
                          cudaMemcpyHostToDevice));

    // Lancement OptiX
    OPTIX_CHECK(optixLaunch(
        pipeline, 0,
        reinterpret_cast<CUdeviceptr>(d_params_buf),
        sizeof(RayGenLaunchParams),
        &sbt,
        num_faces, 1u, 1u));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Récupération des hits
    std::vector<uint8_t> h_hits(num_faces);
    CUDA_CHECK(cudaMemcpy(h_hits.data(), d_hits_buf,
                          num_faces * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));

    // Construction du résultat : distances calculées côté CPU
    // depuis les centroïdes précalculés (évite un transfert GPU → CPU)
    std::vector<float> result(num_faces);

    const float ox = orig.x, oy = orig.y, oz = orig.z;
    for (unsigned int i = 0; i < num_faces; ++i) {
        if (h_hits[i] != 0u) {
            result[i] = -1.0f;  // occultée
        } else {
            const float3& c = h_centroids[i];
            if (std::isnan(c.x)) {
                result[i] = std::numeric_limits<float>::max();  // face dégénérée
            } else {
                float dx = c.x - ox;
                float dy = c.y - oy;
                float dz = c.z - oz;
                result[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
            }
        }
    }

    return result;
}

// ─── traceRay (single point) ─────────────────────────────────────────────────

std::vector<float> RayTracer::traceRay(const Point& p) const
{
    return computeRaysAndHits(p);
}

// ─── traceRay (multiple points) ──────────────────────────────────────────────

std::vector<std::vector<float>> RayTracer::traceRay(const std::vector<Point>& points) const
{
    std::vector<std::vector<float>> result;
    result.reserve(points.size());
    for (const auto& p : points)
        result.push_back(computeRaysAndHits(p));
    return result;
}

// ─── cleanup ─────────────────────────────────────────────────────────────────

void RayTracer::cleanup()
{
    auto freeCU = [](CUdeviceptr& ptr) {
        if (ptr) { cudaFree(reinterpret_cast<void*>(ptr)); ptr = 0; }
    };

    // Buffers persistants de lancement
    if (d_hits_buf)   { cudaFree(d_hits_buf);   d_hits_buf   = nullptr; }
    if (d_params_buf) { cudaFree(d_params_buf); d_params_buf = nullptr; }

    // Buffers de géométrie
    freeCU(d_centroids);
    freeCU(d_vertices);
    freeCU(d_indices);
    freeCU(d_gas_output_buffer);

    // SBT records
    freeCU(raygen_record);
    freeCU(miss_record);
    freeCU(hitgroup_record);

    // Objets OptiX (ordre inverse de création)
    if (pipeline)            { optixPipelineDestroy(pipeline);            pipeline            = nullptr; }
    if (raygen_prog_group)   { optixProgramGroupDestroy(raygen_prog_group);   raygen_prog_group   = nullptr; }
    if (miss_prog_group)     { optixProgramGroupDestroy(miss_prog_group);     miss_prog_group     = nullptr; }
    if (hitgroup_prog_group) { optixProgramGroupDestroy(hitgroup_prog_group); hitgroup_prog_group = nullptr; }
    if (module)              { optixModuleDestroy(module);                module              = nullptr; }
    if (context)             { optixDeviceContextDestroy(context);        context             = nullptr; }

    spdlog::debug("RayTracer resources cleaned up");
}
