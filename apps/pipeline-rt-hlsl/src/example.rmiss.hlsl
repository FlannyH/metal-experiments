struct Payload {
    float3 normal;
    float t;
};

[shader("miss")] void main(inout Payload payload : SV_RayPayload) {}
