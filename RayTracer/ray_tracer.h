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

#include "optix_ray.h"

typedef CGAL::Simple_cartesian<double>  Kernel;
typedef Kernel::Point_3                 Point;
typedef CGAL::Surface_mesh<Point>       SurfaceMesh;

class RayTracer {
public:
    explicit RayTracer(SurfaceMesh& mesh);
    ~RayTracer() = default;

    std::vector<float>              traceRay(const Point& p) const;
    std::vector<std::vector<float>> traceRay(const std::vector<Point>& points) const;

    void cleanup();

private:
    SurfaceMesh& mesh;
    unsigned int num_faces = 0;

    // Centroïdes précalculés (une seule fois au constructeur)
    std::vector<float3> h_centroids;    // côté CPU, pour calcul des distances
    CUdeviceptr         d_centroids = 0; // côté GPU, pour le kernel

    // Buffers GPU persistants (alloués une fois, réutilisés à chaque lancement)
    uint8_t*            d_hits_buf   = nullptr;
    RayGenLaunchParams* d_params_buf = nullptr;

    // OptiX handles
    OptixDeviceContext       context             = nullptr;
    OptixTraversableHandle   gas_handle          = 0;
    OptixModule              module              = nullptr;
    OptixProgramGroup        raygen_prog_group   = nullptr;
    OptixProgramGroup        miss_prog_group     = nullptr;
    OptixProgramGroup        hitgroup_prog_group = nullptr;
    OptixPipeline            pipeline            = nullptr;
    OptixShaderBindingTable  sbt                 = {};

    // SBT device pointers
    CUdeviceptr raygen_record   = 0;
    CUdeviceptr miss_record     = 0;
    CUdeviceptr hitgroup_record = 0;

    // GAS device buffers
    CUdeviceptr d_gas_output_buffer = 0;
    CUdeviceptr d_vertices          = 0;
    CUdeviceptr d_indices           = 0;

    std::vector<float> computeRaysAndHits(const Point& origin) const;
};
