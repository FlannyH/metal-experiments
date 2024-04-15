float4 main(uint vertexIndex : SV_VertexID) : SV_POSITION {
    switch (vertexIndex) {
    case 0:
        return float4(-1, -1, 0, 1);
    case 1:
    case 3:
        return float4(1, -1, 0, 1);
    case 2:
    case 5:
        return float4(-1, 1, 0, 1);
    case 4:
        return float4(1, 1, 0, 1);
    }
    return (0.0 / 0.0).xxxx;
}