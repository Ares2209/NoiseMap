/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#include "ray_tracer.h"
#include "optix_ray.h"

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

// ─── computeRaysAndHits ──────────────────────────────────────────────────────
std::vector<float> RayTracer::computeRaysAndHits(
    const Point&                   origin,
    OptixDeviceContext              optixContext,
    OptixTraversableHandle          gasHandle,
    OptixPipeline                   pipelineArg,
    const OptixShaderBindingTable&  sbtArg) const
{
    // ── Build centroid list ──────────────────────────────────────────────────
    std::vector<Point> centroids;
    centroids.reserve(mesh.number_of_faces());

    for (auto f : mesh.faces()) {
        std::vector<Point> tri;
        tri.reserve(3);
        for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh))
            tri.push_back(mesh.point(v));

        if (tri.size() == 3)
            centroids.push_back(CGAL::centroid(tri[0], tri[1], tri[2]));
    }

    const size_t numRays = centroids.size();

    // ── Build ray list ───────────────────────────────────────────────────────
    std::vector<OptixRay> rays;
    std::vector<float>    distances;
    rays.reserve(numRays);
    distances.reserve(numRays);

    const float3 orig = make_float3(
        static_cast<float>(origin.x()),
        static_cast<float>(origin.y()),
        static_cast<float>(origin.z()));

    for (const auto& c : centroids) {
        float dx = static_cast<float>(c.x() - origin.x());
        float dy = static_cast<float>(c.y() - origin.y());
        float dz = static_cast<float>(c.z() - origin.z());

        float len = sqrtf(dx*dx + dy*dy + dz*dz);

        if (len < 1e-6f) {
            distances.push_back(0.0f);
            rays.push_back({ orig, make_float3(0.f, 0.f, 1.f), 0.0f });
            continue;
        }

        float3 dir = make_float3(dx / len, dy / len, dz / len);

        distances.push_back(len);

        // tmin : petit offset pour éviter l'auto-intersection à la source
        // tmax : 99% de la distance → on n'atteint pas la face cible elle-même
        //        si un triangle est touché avant → c'est une vraie occlusion
        rays.push_back({ orig, dir, len * 0.99f });
    }

    // ── Device allocations ───────────────────────────────────────────────────
    OptixRay* d_rays = nullptr;
    uint8_t*  d_hits = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rays, numRays * sizeof(OptixRay)));
    CUDA_CHECK(cudaMalloc(&d_hits, numRays * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_hits, 0, numRays * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_rays, rays.data(),
                          numRays * sizeof(OptixRay),
                          cudaMemcpyHostToDevice));

    // ── Launch params ────────────────────────────────────────────────────────
    RayGenLaunchParams params;
    params.rays      = d_rays;
    params.hits      = d_hits;
    params.numRays   = static_cast<unsigned int>(numRays);
    params.gasHandle = gasHandle;

    RayGenLaunchParams* d_params = nullptr;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(RayGenLaunchParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &params,
                          sizeof(RayGenLaunchParams),
                          cudaMemcpyHostToDevice));

    // ── OptiX launch ─────────────────────────────────────────────────────────
    OPTIX_CHECK(optixLaunch(
        pipelineArg,
        static_cast<CUstream>(0),
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(RayGenLaunchParams),
        &sbtArg,
        static_cast<unsigned int>(numRays), 1u, 1u));

    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Copy results ─────────────────────────────────────────────────────────
    std::vector<uint8_t> hits(numRays, 0u);
    CUDA_CHECK(cudaMemcpy(hits.data(), d_hits,
                          numRays * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_rays));
    CUDA_CHECK(cudaFree(d_hits));
    CUDA_CHECK(cudaFree(d_params));

    // ── Appliquer l'occlusion ─────────────────────────────────────────────────
    // hits[i] == 1 → un triangle a été touché AVANT le centroïd cible
    //            → la face est occultée → distance = max
    // hits[i] == 0 → aucun obstacle     → face visible → distance conservée
    for (size_t i = 0; i < numRays; ++i) {
        if (hits[i] != 0u)
            distances[i] = std::numeric_limits<float>::max();
    }

    return distances;
}


// ─── Constructor ─────────────────────────────────────────────────────────────

RayTracer::RayTracer(SurfaceMesh& mesh) : mesh(mesh)
{
    spdlog::debug("Initializing RayTracer: {} vertices, {} faces",
                  mesh.number_of_vertices(), mesh.number_of_faces());

    // 1. CUDA + OptiX init ────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(nullptr));          // force CUDA context creation
    OPTIX_CHECK(optixInit());

    // 2. Device context ───────────────────────────────────────────────────────
    OptixDeviceContextOptions ctxOpts = {};
    ctxOpts.logCallbackFunction       = &optixLogger;
    ctxOpts.logCallbackLevel          = 4;
#ifndef NDEBUG
    ctxOpts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    ctxOpts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &ctxOpts, &context));

    // 3. Upload geometry ──────────────────────────────────────────────────────
    std::vector<float3>       vertices;
    std::vector<uint3>        triangles;
    vertices.reserve(mesh.number_of_vertices());
    triangles.reserve(mesh.number_of_faces());

    for (auto v : mesh.vertices()) {
        auto p = mesh.point(v);
        vertices.push_back(make_float3(
            static_cast<float>(p.x()),
            static_cast<float>(p.y()),
            static_cast<float>(p.z())));
    }

    for (auto f : mesh.faces()) {
        std::vector<unsigned int> idx;
        idx.reserve(3);
        for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh))
            idx.push_back(static_cast<unsigned int>(v));
        if (idx.size() == 3)
            triangles.push_back(make_uint3(idx[0], idx[1], idx[2]));
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices),
                          vertices.size()  * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices),
                          triangles.size() * sizeof(uint3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices),
                          vertices.data(),
                          vertices.size() * sizeof(float3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices),
                          triangles.data(),
                          triangles.size() * sizeof(uint3),
                          cudaMemcpyHostToDevice));

    // 4. Build input ──────────────────────────────────────────────────────────
    OptixBuildInput buildInput                           = {};
    buildInput.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes         = sizeof(float3);
    buildInput.triangleArray.numVertices                 = static_cast<unsigned>(vertices.size());
    buildInput.triangleArray.vertexBuffers               = &d_vertices;
    buildInput.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes          = sizeof(uint3);
    buildInput.triangleArray.numIndexTriplets            = static_cast<unsigned>(triangles.size());
    buildInput.triangleArray.indexBuffer                 = d_indices;

    static const unsigned int s_geom_flags[1]            = { OPTIX_GEOMETRY_FLAG_NONE };
    buildInput.triangleArray.flags                        = s_geom_flags;
    buildInput.triangleArray.numSbtRecords                = 1;

    // 5. Build GAS ────────────────────────────────────────────────────────────
    OptixAccelBuildOptions accelOpts     = {};
    accelOpts.buildFlags                 = OPTIX_BUILD_FLAG_NONE;
    accelOpts.operation                  = OPTIX_BUILD_OPERATION_BUILD;
    // motionOptions is zero-initialised → numKeys defaults to 0
    // which means "no motion blur" (valid for static geometry)

    OptixAccelBufferSizes gasSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context, &accelOpts, &buildInput, 1, &gasSizes));

    CUdeviceptr d_temp = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp),
                          gasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer),
                          gasSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context,
        static_cast<CUstream>(0),
        &accelOpts,
        &buildInput, 1,
        d_temp,                  gasSizes.tempSizeInBytes,
        d_gas_output_buffer,     gasSizes.outputSizeInBytes,
        &gas_handle,
        nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());   // ensure GAS is built before freeing temp
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));

    // 6. Module ───────────────────────────────────────────────────────────────
    OptixPipelineCompileOptions pipelineCompileOpts  = {};
    pipelineCompileOpts.usesMotionBlur               = false;
    pipelineCompileOpts.traversableGraphFlags        =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOpts.numPayloadValues             = 1;
    pipelineCompileOpts.numAttributeValues           = 2;
    pipelineCompileOpts.exceptionFlags               = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOpts.pipelineLaunchParamsVariableName = "params";

    OptixModuleCompileOptions moduleCompileOpts = {};
    moduleCompileOpts.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
    moduleCompileOpts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
    moduleCompileOpts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    char   log[4096];
    size_t logSize = sizeof(log);

    spdlog::trace("Loading OptiX IR/PTX from RayTracer.optixir");
    std::ifstream ptxFile("RayTracer.optixir", std::ios::binary);
    if (!ptxFile.is_open())
        spdlog::error("Cannot open RayTracer.optixir");

    std::string ptx((std::istreambuf_iterator<char>(ptxFile)),
                     std::istreambuf_iterator<char>());

    OPTIX_CHECK(optixModuleCreate(
        context,
        &moduleCompileOpts, &pipelineCompileOpts,
        ptx.c_str(), ptx.size(),
        log, &logSize,
        &module));
    if (logSize > 1) spdlog::trace("Module log: {}", log);

    // 7. Program groups ───────────────────────────────────────────────────────
    OptixProgramGroupOptions pgOpts = {};

    // Raygen
    {
        OptixProgramGroupDesc desc     = {};
        desc.kind                      = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module             = module;
        desc.raygen.entryFunctionName  = "__raygen__rg";
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context, &desc, 1, &pgOpts, log, &logSize, &raygen_prog_group));
        if (logSize > 1) spdlog::trace("Raygen PG log: {}", log);
    }

    // Miss
    {
        OptixProgramGroupDesc desc   = {};
        desc.kind                    = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module             = module;
        desc.miss.entryFunctionName  = "__miss__ms";
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context, &desc, 1, &pgOpts, log, &logSize, &miss_prog_group));
        if (logSize > 1) spdlog::trace("Miss PG log: {}", log);
    }

    // Hitgroup
    {
        OptixProgramGroupDesc desc                    = {};
        desc.kind                                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH                        = module;
        desc.hitgroup.entryFunctionNameCH             = "__closesthit__ch";
        desc.hitgroup.moduleAH                        = nullptr;
        desc.hitgroup.entryFunctionNameAH             = nullptr;
        desc.hitgroup.moduleIS                        = nullptr;
        desc.hitgroup.entryFunctionNameIS             = nullptr;
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context, &desc, 1, &pgOpts, log, &logSize, &hitgroup_prog_group));
        if (logSize > 1) spdlog::trace("Hitgroup PG log: {}", log);
    }

    // 8. Pipeline ─────────────────────────────────────────────────────────────
    OptixProgramGroup pgs[] = {
        raygen_prog_group,
        miss_prog_group,
        hitgroup_prog_group
    };

    OptixPipelineLinkOptions linkOpts = {};
    linkOpts.maxTraceDepth            = 1;

    logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        context,
        &pipelineCompileOpts,
        &linkOpts,
        pgs, 3,
        log, &logSize,
        &pipeline));
    if (logSize > 1) spdlog::trace("Pipeline log: {}", log);

    // ── Stack sizes (compatible OptiX 7.x) ───────────────────────────────────
    {
        OptixStackSizes ss          = {};
        unsigned int    css_rg      = 0u;
        unsigned int    css_ms      = 0u;
        unsigned int    css_ch      = 0u;

        OPTIX_CHECK(optixProgramGroupGetStackSize(
            raygen_prog_group, &ss, pipeline));
        css_rg = ss.cssRG;

        OPTIX_CHECK(optixProgramGroupGetStackSize(
            miss_prog_group, &ss, pipeline));
        css_ms = ss.cssMS;

        OPTIX_CHECK(optixProgramGroupGetStackSize(
            hitgroup_prog_group, &ss, pipeline));
        css_ch = ss.cssCH;

        // Pour maxTraceDepth = 1 et aucun callable :
        unsigned int continuationStack =
            std::max(css_rg, css_ms + css_ch * linkOpts.maxTraceDepth);

        OPTIX_CHECK(optixPipelineSetStackSize(
            pipeline,
            0u,                 // direct callable depuis traversal
            0u,                 // direct callable depuis état
            continuationStack,
            1u));               // profondeur max du graphe de traversal

        spdlog::debug(
            "Stack sizes — cssRG:{} cssMS:{} cssCH:{} → continuation:{}",
            css_rg, css_ms, css_ch, continuationStack);
    }

    // 9. SBT ──────────────────────────────────────────────────────────────────
    const size_t recSize = OPTIX_SBT_RECORD_HEADER_SIZE;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record),   recSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record),     recSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), recSize));

    {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char buf[OPTIX_SBT_RECORD_HEADER_SIZE];
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, buf));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record),
                              buf, recSize, cudaMemcpyHostToDevice));
    }
    {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char buf[OPTIX_SBT_RECORD_HEADER_SIZE];
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, buf));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record),
                              buf, recSize, cudaMemcpyHostToDevice));
    }
    {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char buf[OPTIX_SBT_RECORD_HEADER_SIZE];
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, buf));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record),
                              buf, recSize, cudaMemcpyHostToDevice));
    }

    sbt = {};
    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = static_cast<unsigned int>(recSize);
    sbt.missRecordCount             = 1u;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(recSize);
    sbt.hitgroupRecordCount         = 1u;

    spdlog::debug("RayTracer initialized successfully");

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
