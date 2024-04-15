#include "breda-gpu-shared::math.hlsl"
#include "breda-render-backend-api::bindless.hlsl"

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR;
};

struct Vertex {
    float2 position;
    float2 texCoord;
    uint color;
};

struct Bindings {
    float2 resolution;
    RawBuffer colorSpaceBindings;
#ifndef RENDER_SINGLE
    ArrayBuffer vertexData;
#else
    float4 renderArea;
#endif
};

VertexOutput main(uint vertexIndex : SV_VertexID) {
    const Bindings bnd = loadBindings<Bindings>();

#ifndef RENDER_SINGLE

    Vertex vertex = bnd.vertexData.loadUniform<Vertex>(vertexIndex);
    const float2 position = vertex.position;
    const float2 texCoord = vertex.texCoord;
    const float4 color = float4((vertex.color >> 0) & 0xff, (vertex.color >> 8) & 0xff,
                                (vertex.color >> 16) & 0xff, (vertex.color >> 24) & 0xff) /
                         255.0;

#else

    float2 position, texCoord;
    const float4 color = float4(1.0, 1.0, 1.0, 1.0);

    switch (vertexIndex) {
    case 0:
        position = bnd.renderArea.xy;
        texCoord = float2(0.0, 0.0);
        break;
    case 1:
    case 3:
        position = bnd.renderArea.zy;
        texCoord = float2(1.0, 0.0);
        break;
    case 2:
    case 5:
        position = bnd.renderArea.xw;
        texCoord = float2(0.0, 1.0);
        break;
    case 4:
        position = bnd.renderArea.zw;
        texCoord = float2(1.0, 1.0);
        break;
    default:
        position = (0.0 / 0.0).xx;
        texCoord = (0.0 / 0.0).xx;
        break;
    }

#endif

    VertexOutput o;
    o.position = float4(uvToClip(position / bnd.resolution), 0.0, 1.0);
    o.texCoord = texCoord;
    o.color = color;
    return o;
}
