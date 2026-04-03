/*
 * Created on Fri Oct 03 2025
 *
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0 (the 'License')
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * **Additional Restriction**: This code may not be used for commercial purposes.
 */


#include <optix.h>
#include "optix_ray.h"

#include <cstdio>

extern "C" {
__constant__ RayGenLaunchParams params;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.numRays) return;

    OptixRay ray = params.rays[idx.x];
    unsigned int hit = 0;

    optixTrace(
        params.gasHandle,
        ray.origin,
        ray.direction,
        0.001f,    // tmin
        ray.tmax,   // tmax
        0.0f,    // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT record
        hit
    );
    params.hits[idx.x] = (hit != 0);
}

extern "C" __global__ void __closesthit__ch()
{   
    optixSetPayload_0(1); 
}
extern "C" __global__ void __miss__ms()
{
    printf("Miss function called : This should not happened");
}