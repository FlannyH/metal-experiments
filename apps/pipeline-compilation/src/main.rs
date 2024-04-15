use std::ffi::CString;

use libloading::Library;
use saxaboom::{
    IRComparisonFunction, IRCompiler, IRFilter, IRMetalLibBinary, IRObject, IRRootConstants,
    IRRootParameter1, IRRootParameter1_u, IRRootParameterType, IRRootSignature,
    IRRootSignatureDescriptor1, IRRootSignatureFlags, IRRootSignatureVersion, IRShaderStage,
    IRShaderVisibility, IRStaticBorderColor, IRStaticSamplerDescriptor, IRTextureAddressMode,
    IRVersionedRootSignatureDescriptor, IRVersionedRootSignatureDescriptor_u,
};

fn main() {
    let device = metal::Device::system_default().unwrap();

    // Get IRCompiler
    let lib = unsafe { libloading::Library::new("lib/libmetalirconverter.dylib") }.unwrap();

    // Load DXIL and convert to metallib
    let vs_dxil = include_bytes!("egui_render.vs.dxil");
    let ps_dxil = include_bytes!("egui_render.ps.dxil");
    let vs_metallib =
        compile_dxil_to_metallib(&lib, vs_dxil, "main", IRShaderStage::IRShaderStageVertex)
            .unwrap();
    let ps_metallib =
        compile_dxil_to_metallib(&lib, ps_dxil, "main", IRShaderStage::IRShaderStageFragment)
            .unwrap();
    let vs_library = device.new_library_with_data(&vs_metallib).unwrap();
    let ps_library = device.new_library_with_data(&ps_metallib).unwrap();
    let vs_func = vs_library.get_function("main", None).unwrap();
    let ps_func = ps_library.get_function("main", None).unwrap();

    // Create pipeline from metallib
    let pipeline_desc = metal::RenderPipelineDescriptor::new();
    dbg!(&vs_func);
    dbg!(&ps_func);
    pipeline_desc.set_vertex_function(Some(&vs_func));
    pipeline_desc.set_fragment_function(Some(&ps_func));

    // crash?
    device.new_render_pipeline_state(&pipeline_desc).unwrap();
    println!("If you see this, it worked!");
    dbg!(pipeline_desc);
}

fn compile_dxil_to_metallib(
    metal_irconverter: &Library,
    dxil_binary: &[u8],
    entry_point: &str,
    shader_type: IRShaderStage,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Load the metal shader converter library
    let lib = metal_irconverter;

    // Set up root signature. This should match up with the root signature from dx12
    let parameters = {
        let push_constants = IRRootParameter1 {
            parameter_type: IRRootParameterType::IRRootParameterType32BitConstants,
            shader_visibility: IRShaderVisibility::IRShaderVisibilityAll,
            u: IRRootParameter1_u {
                constants: IRRootConstants {
                    register_space: 0,
                    shader_register: 0,
                    num32_bit_values: 4, // debug has 6
                },
            },
        };

        let indirect_identifier = IRRootParameter1 {
            parameter_type: IRRootParameterType::IRRootParameterType32BitConstants,
            shader_visibility: IRShaderVisibility::IRShaderVisibilityAll,
            u: IRRootParameter1_u {
                constants: IRRootConstants {
                    register_space: 1,
                    shader_register: 0,
                    num32_bit_values: 1,
                },
            },
        };

        vec![push_constants, indirect_identifier]
    };

    let static_samplers = [
        create_static_sampler(
            IRFilter::IRFilterMinMagMipPoint,
            IRTextureAddressMode::IRTextureAddressModeWrap,
            0,
            None,
        ),
        create_static_sampler(
            IRFilter::IRFilterMinMagMipPoint,
            IRTextureAddressMode::IRTextureAddressModeClamp,
            1,
            None,
        ),
        create_static_sampler(
            IRFilter::IRFilterMinMagMipLinear,
            IRTextureAddressMode::IRTextureAddressModeWrap,
            2,
            None,
        ),
        create_static_sampler(
            IRFilter::IRFilterMinMagMipLinear,
            IRTextureAddressMode::IRTextureAddressModeClamp,
            3,
            None,
        ),
        create_static_sampler(
            IRFilter::IRFilterMinMagMipLinear,
            IRTextureAddressMode::IRTextureAddressModeBorder,
            4,
            None,
        ),
        create_static_sampler(
            IRFilter::IRFilterAnisotropic,
            IRTextureAddressMode::IRTextureAddressModeWrap,
            5,
            Some(2),
        ),
        create_static_sampler(
            IRFilter::IRFilterAnisotropic,
            IRTextureAddressMode::IRTextureAddressModeWrap,
            6,
            Some(4),
        ),
    ];

    let desc_1_1 = IRRootSignatureDescriptor1 {
        flags: IRRootSignatureFlags::IRRootSignatureFlagCBVSRVUAVHeapDirectlyIndexed,
        num_parameters: parameters.len() as u32,
        p_parameters: parameters.as_ptr(),
        num_static_samplers: static_samplers.len() as u32,
        p_static_samplers: static_samplers.as_ptr(),
    };

    let desc = IRVersionedRootSignatureDescriptor {
        version: IRRootSignatureVersion::IRRootSignatureVersion_1_1,
        u: IRVersionedRootSignatureDescriptor_u { desc_1_1 },
    };

    let root_sig = IRRootSignature::create_from_descriptor(lib, &desc)?;

    // Cross-compile to Metal
    let mut mtl_binary = IRMetalLibBinary::new(lib)?;
    let obj = IRObject::create_from_dxil(lib, dxil_binary)?;
    let mut c = IRCompiler::new(lib)?;
    c.set_global_root_signature(&root_sig);

    let entry_point_cstring = CString::new(entry_point).unwrap();

    let mtllib = c.alloc_compile_and_link(&entry_point_cstring, &obj)?;
    mtllib.get_metal_lib_binary(shader_type, &mut mtl_binary);

    Ok(mtl_binary.get_byte_code())
}

fn create_static_sampler(
    min_mag_mip_mode: IRFilter,
    address_mode: IRTextureAddressMode,
    index: u32,
    anisotropy: Option<u32>,
) -> IRStaticSamplerDescriptor {
    let max_anisotropy = anisotropy.unwrap_or(1);

    IRStaticSamplerDescriptor {
        filter: min_mag_mip_mode,
        address_u: address_mode,
        address_v: address_mode,
        address_w: address_mode,
        mip_lod_bias: 0.0,
        max_anisotropy,
        comparison_func: IRComparisonFunction::IRComparisonFunctionNever,
        min_lod: 0.0,
        max_lod: 100000.0,
        shader_register: index,
        register_space: 0,
        shader_visibility: IRShaderVisibility::IRShaderVisibilityAll,
        border_color: IRStaticBorderColor::IRStaticBorderColorTransparentBlack,
    }
}
