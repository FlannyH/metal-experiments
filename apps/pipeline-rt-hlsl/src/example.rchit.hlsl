struct Payload {
    float3 normal;
    float t;
};

struct IntersectAttributes {
    float2 barycentrics;
};

[shader("closesthit")] void main(inout Payload payload
                                 : SV_RayPayload, IntersectAttributes attribs) {
    payload.t = RayTCurrent();
    payload.normal = normalize(float3(0, 0, -1));
}
