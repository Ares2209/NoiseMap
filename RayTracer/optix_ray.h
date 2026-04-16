/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <cstdint>

struct RayGenLaunchParams {
    float3                 origin;        // source commune à tous les rayons
    float3*                centroids;     // centroïdes des faces (GPU, précalculés)
    uint8_t*               hits;          // sortie : 1 = occultée, 0 = visible
    unsigned int           numRays;
    OptixTraversableHandle gasHandle;
    float                  tmaxOffsetRel; // fraction de la distance (défaut 0.01)
    float                  tmaxOffsetMin; // offset minimum (défaut 1e-3)
    float                  tmaxOffsetMax; // offset maximum (défaut 0.5)
};
