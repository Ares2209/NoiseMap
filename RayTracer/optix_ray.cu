#include <optix.h>
#include "optix_ray.h"

extern "C" {
__constant__ RayGenLaunchParams params;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.numRays) return;

    const float3 centroid = params.centroids[idx.x];
    const float3 orig     = params.origin;

    // Direction vers le centroïde
    float dx  = centroid.x - orig.x;
    float dy  = centroid.y - orig.y;
    float dz  = centroid.z - orig.z;
    float len = sqrtf(dx*dx + dy*dy + dz*dz);

    // Face dégénérée ou source confondue avec le centroïde
    if (len < 1e-6f || isnan(centroid.x)) {
        params.hits[idx.x] = 0u;
        return;
    }

    float inv_len = 1.0f / len;
    float3 dir = make_float3(dx * inv_len, dy * inv_len, dz * inv_len);

    // tmax adaptatif : s'arrêter avant d'atteindre la face cible
    float offset = fminf(fmaxf(params.tmaxOffsetRel * len, params.tmaxOffsetMin),
                         params.tmaxOffsetMax);
    float tmax = len - offset;

    unsigned int hitFlag   = 0u;
    unsigned int hitFaceId = 0xFFFFFFFFu;

    optixTrace(
        params.gasHandle,
        orig,
        dir,
        1e-3f,          // tmin
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        hitFlag,
        hitFaceId
    );

    // idx.x == face index (un rayon par face, dans l'ordre)
    const bool occluded = (hitFlag != 0u) && (hitFaceId != idx.x);
    params.hits[idx.x] = occluded ? 1u : 0u;
}

extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(optixGetPrimitiveIndex());
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0u);
    optixSetPayload_1(0xFFFFFFFFu);
}
