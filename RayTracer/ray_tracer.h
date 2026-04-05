/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#pragma once

#include <vector>
#include <optix.h>
#include <cuda_runtime.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

using namespace std;

typedef CGAL::Simple_cartesian<double>  Kernel;
typedef Kernel::Point_3                 Point;
typedef CGAL::Surface_mesh<Point>       SurfaceMesh;

class RayTracer {
public:
    explicit RayTracer(SurfaceMesh& mesh);
    ~RayTracer() = default;

    vector<float>         traceRay(const Point& p) const;
    vector<float>         traceRay(const Point& p,
                                   const vector<Point>& centroids) const;
    vector<vector<float>> traceRay(const vector<Point>& points) const;

    void cleanup();

private:
    SurfaceMesh& mesh;

    // OptiX handles
    mutable OptixDeviceContext   context          = nullptr;
    mutable OptixTraversableHandle gas_handle      = 0;
    mutable OptixModule          module           = nullptr;
    mutable OptixProgramGroup    raygen_prog_group = nullptr;
    mutable OptixProgramGroup    miss_prog_group   = nullptr;
    mutable OptixProgramGroup    hitgroup_prog_group = nullptr;
    mutable OptixPipeline        pipeline         = nullptr;
    mutable OptixShaderBindingTable sbt           = {};

    // SBT device pointers
    CUdeviceptr raygen_record   = 0;
    CUdeviceptr miss_record     = 0;
    CUdeviceptr hitgroup_record = 0;

    // GAS device buffers (kept alive for the lifetime of the object)
    CUdeviceptr d_gas_output_buffer = 0;
    CUdeviceptr d_vertices          = 0;
    CUdeviceptr d_indices           = 0;

    vector<float> computeRaysAndHits(
        const Point&             origin,
        OptixDeviceContext        optixContext,
        OptixTraversableHandle   gasHandle,
        OptixPipeline            pipeline,
        const OptixShaderBindingTable& sbt) const;

    vector<float> computeRaysFromCentroids(
        const Point&             origin,
        const vector<Point>&     centroids,
        OptixDeviceContext        optixContext,
        OptixTraversableHandle   gasHandle,
        OptixPipeline            pipeline,
        const OptixShaderBindingTable& sbt) const;
};
