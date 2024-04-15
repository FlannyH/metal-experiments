#include "breda-color-space::color_space_conversions.hlsl"
#include "breda-render-backend-api::bindless.hlsl"

#define TEXTURE_FILTER_LINEAR 0
#define TEXTURE_FILTER_POINT 1
#define PREAPPLIED_COLOR_SPACE_TRANSFORM 1

struct Bindings {
    float2 resolution;
    RawBuffer colorSpaceBindings;
};

struct VertexInput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR;
};

float4 main(VertexInput input) : SV_Target0 {
    Bindings bnd = loadBindings<Bindings>();

    Texture texture = Texture::unsafeFromUint(g_bindingsOffset.userData0);
    uint filterMode = g_bindingsOffset.userData1;
    uint spaceMode = g_bindingsOffset.userData2;

    float4 color;
    switch (filterMode) {
    case TEXTURE_FILTER_LINEAR:
        color = texture.sample2DUniform<float4>(samplerMinMagMipLinearClamp(), input.texCoord);
        break;
    case TEXTURE_FILTER_POINT:
        color = texture.sample2DUniform<float4>(samplerMinMagMipPointClamp(), input.texCoord);
        break;
    default:
        // The possible sample modes should always be up to date, so this is never the case.
        ASSERT(false);
        color = (0.0 / 0.0).xxxx;
        break;
    }

    color = color * input.color;

    if (bnd.colorSpaceBindings.isValid() && spaceMode != PREAPPLIED_COLOR_SPACE_TRANSFORM) {
        color = applyClf(acesCgToAcesCct(applyIdt(color)), bnd.colorSpaceBindings);
    }

    return color;
}
