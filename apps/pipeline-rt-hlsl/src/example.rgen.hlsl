#include "breda-render-backend-api::bindless.hlsl"

struct Bindings {
    UniformAccelerationStructure tlas;
    RwTexture output;
};

struct Payload {
    float3 normal;
    float t;
};

[shader("raygeneration")] void main() {
    Bindings bnd = loadBindings<Bindings>();

    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 pixelCenter = launchIndex + 0.5;
    float3 wsPos = float3(pixelCenter, -1);

    Payload payload = (Payload)0;
    payload.t = -1.0;

    RayDesc ray;
    ray.Origin = wsPos;
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.1;
    ray.TMax = 1000.0;

    float3 T = 0.0;

    TraceRay(bnd.tlas.topLevelTemporary(), RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);

    if (payload.t > 0.0) {
        T += abs(payload.normal);
    } else {
        T += float3(0, 1, 0);
    }

    bnd.output.store2DUniform<float4>(launchIndex, float4(T, 1));
}
