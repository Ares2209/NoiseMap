/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <cstdint>                  // uint8_t

#pragma once
#include <optix.h>
#include <cuda_runtime.h>

struct OptixRay {
    float3       origin;
    float3       direction;
    float        tmax;
    unsigned int targetFaceIdx;   // index de la face cible dans le GAS
};

struct RayGenLaunchParams {
    OptixRay*              rays;
    uint8_t*               hits;       // 1 = occultée, 0 = visible
    unsigned int           numRays;
    OptixTraversableHandle gasHandle;
};
