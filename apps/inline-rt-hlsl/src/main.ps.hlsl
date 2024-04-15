#include "breda-render-backend-api::bindless.hlsl"

struct Bindings {
    UniformAccelerationStructure tlas;
};

float4 main(float4 input : SV_POSITION) : SV_Target0 {
    uint2 launchIndex = input.xy;

    Bindings bnd = loadBindings<Bindings>();

    float2 pixelCenter = launchIndex + 0.5f;

    float3 wsPos = float3(pixelCenter, -1);

    RayDesc ray;
    ray.Origin = wsPos;
    ray.TMin = 0.1;
    ray.TMax = 1000.0;
    ray.Direction = float3(0, 0, 1);

    float3 T = 0.0f;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(bnd.tlas.topLevelTemporary(), 0, 0xff, ray);
    q.Proceed(); // No looping needed thanks to ACCEPT_FIRST_HIT

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        T += float3(1, 0, 1);
    } else {
        T += float3(0, 1, 0);
    }

    return float4(T, 1.0f);
}
