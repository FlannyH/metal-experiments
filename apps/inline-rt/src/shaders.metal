#include <metal_stdlib>
#include <metal_raytracing>

using namespace metal;
using namespace raytracing;

struct VertexShaderOutput {
    float4 position [[position]];
};

vertex VertexShaderOutput v_main(const device packed_float2* vertices [[buffer(0)]], uint vert_id [[vertex_id]]) {
    VertexShaderOutput out;
    out.position = float4(vertices[vert_id].x, vertices[vert_id].y, 0.0, 1.0);
    return out;
}

fragment float4 f_main(VertexShaderOutput input [[stage_in]], instance_acceleration_structure acc [[buffer(0)]]) {
    // Create ray
    float2 pixel_center = input.position.xy + 0.5f;
    float3 ws_pos = float3(pixel_center, -1);
    ray r;
    r.origin = ws_pos;
    r.min_distance = 0.1;
    r.max_distance = 1000.0;
    r.direction = float3(0, 0, 1);

    intersector<triangle_data, instancing> i;

    i.force_opacity(forced_opacity::opaque);
    i.accept_any_intersection(true);

    auto result = i.intersect(r, acc);
    if (result.type == intersection_type::triangle) {
        return float4(1.0, 0.0, 1.0, 1.0);
    } else {
        return float4(0.0, 1.0, 0.0, 1.0);
    }
    return float4(input.position.x / 10000, 0.0, 0.0, 1.0);
}