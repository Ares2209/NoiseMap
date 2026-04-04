#include <optix.h>
#include "optix_ray.h"

extern "C" {
__constant__ RayGenLaunchParams params;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.numRays) return;

    const OptixRay ray = params.rays[idx.x];

    // payload 0 : hit flag
    // payload 1 : index du triangle touché (pour filtrer la face cible)
    unsigned int hitFlag   = 0u;
    unsigned int hitFaceId = 0xFFFFFFFFu;

    optixTrace(
        params.gasHandle,
        ray.origin,
        ray.direction,
        1e-3f,          // tmin : offset pour éviter self-intersection à la source
        ray.tmax,       // tmax : fixé côté CPU avec marge suffisante
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        hitFlag,        // payload 0
        hitFaceId       // payload 1
    );

    const bool occluded = (hitFlag != 0u) 
                       && (hitFaceId != ray.targetFaceIdx);

    params.hits[idx.x] = occluded ? 1u : 0u;
}

extern "C" __global__ void __closesthit__ch()
{
    // Index du triangle dans le GAS = index dans le tableau de faces original
    optixSetPayload_0(1u);
    optixSetPayload_1(optixGetPrimitiveIndex());
}

extern "C" __global__ void __miss__ms()
{
    // Aucun triangle touché → face visible
    optixSetPayload_0(0u);
    optixSetPayload_1(0xFFFFFFFFu);
}
