use std::ffi::{c_void, CString};
use std::io::Write;
use std::mem::{self, size_of};
use std::time::Instant;

use cocoa::appkit::NSView;
use metal::foreign_types::ForeignType;
use metal::{
    Device, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderStages,
    MTLResourceOptions, MTLResourceUsage, MTLStoreAction, MetalLayer, RenderPassDescriptor,
};
use objc::rc::autoreleasepool;
use objc::runtime::YES;
use saxaboom::{
    IRComparisonFunction, IRCompiler, IRFilter, IRMetalLibBinary, IRObject, IRRootConstants,
    IRRootParameter1, IRRootParameter1_u, IRRootParameterType, IRRootSignature,
    IRRootSignatureDescriptor1, IRRootSignatureFlags, IRRootSignatureVersion, IRShaderReflection,
    IRShaderStage, IRShaderVisibility, IRStaticBorderColor, IRStaticSamplerDescriptor,
    IRTextureAddressMode, IRVersionedRootSignatureDescriptor, IRVersionedRootSignatureDescriptor_u,
};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::platform::macos::WindowExtMacOS;

const BIND_POINT_DESCRIPTOR_HEAP: u64 = 0;
const _BIND_POINT_SAMPLER_HEAP: u64 = 1;
const BIND_POINT_ARGUMENT_BUFFER: u64 = 2;
const _BIND_POINT_ARGUMENT_BUFFER_HULL_DOMAIN: u64 = 3;
const BIND_POINT_RAY_DISPATCH_ARGUMENTS: u64 = 3;
const BIND_POINT_ARGUMENT_BUFFER_DRAW_ARGUMENTS: u64 = 4;
const BIND_POINT_ARGUMENT_BUFFER_UNIFORMS: u64 = 5;
const _BIND_POINT_VERTEX_BUFFER: u64 = 6;

fn main() {
    // Create a window
    let event_loop = winit::event_loop::EventLoop::new();
    let res = winit::dpi::LogicalSize::new(1280, 720);
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(res)
        .with_title("RustRenderMetal".to_string())
        .build(&event_loop)
        .unwrap();

    // Get device
    let device = Device::system_default().expect("Could not create device.");

    // Create metal layer
    let layer = MetalLayer::new();
    layer.set_device(&device);
    layer.set_pixel_format(MTLPixelFormat::RGBA16Float);
    layer.set_presents_with_transaction(false);
    layer.set_framebuffer_only(false);

    // Create view
    unsafe {
        let view = window.ns_view() as cocoa::base::id;
        view.setWantsLayer(YES);
        view.setLayer(mem::transmute(layer.as_ptr()));
    }

    // Create resource heaps
    let heap_shared = {
        let heap_desc_shared = metal::HeapDescriptor::new();
        heap_desc_shared.set_size(16 * 1024 * 1024);
        heap_desc_shared.set_storage_mode(metal::MTLStorageMode::Shared);
        device.new_heap(&heap_desc_shared)
    };
    let heap_private = {
        let heap_desc_private = metal::HeapDescriptor::new();
        heap_desc_private.set_size(16 * 1024 * 1024);
        heap_desc_private.set_storage_mode(metal::MTLStorageMode::Private);
        device.new_heap(&heap_desc_private)
    };

    // Commands
    let command_queue = device.new_command_queue();

    // Create vertex buffer
    let positions = vec![
        [100.0f32, 100.1f32, 0.0f32],
        [200.0f32, 100.2f32, 0.0f32],
        [302.0f32, 403.0f32, 0.0f32],
    ];
    let options = MTLResourceOptions::StorageModeShared;
    let vertex_buffer = new_buffer_with_data(&heap_shared, positions, options, "vertex buffer");

    // Create index buffer
    let indices = vec![0u32, 1, 2, 3, 4, 5];
    let index_buffer = new_buffer_with_data(&heap_shared, indices, options, "index buffer");

    let (blas, tlas) = build_acceleration_structure(
        &heap_shared,
        &heap_private,
        vertex_buffer,
        &index_buffer,
        &device,
        &command_queue,
    );

    // Make instance contributions buffer
    let instance_contributions = vec![0u32];
    let instance_contributions_buffer = new_buffer_with_data(
        &heap_shared,
        instance_contributions,
        MTLResourceOptions::StorageModeShared,
        "instance contributions buffer",
    );

    // Make acceleration structure GPU header
    let acc_structure_gpu_header = RaytracingAccelerationStructureGPUHeader {
        acceleration_structure_id: tlas.gpu_resource_id()._impl,
        address_of_instance_contributions: instance_contributions_buffer.gpu_address(),
        ..Default::default()
    };
    let acc_structure_gpu_header_buffer = new_buffer_with_data(
        &heap_shared,
        vec![acc_structure_gpu_header],
        MTLResourceOptions::StorageModeShared,
        "acceleration structure gpu header",
    );

    // Make bindings buffer
    let bindings = vec![create_render_resource_handle(
        0,
        RenderResourceTag::Buffer,
        2,
    )];
    let bindings_buffer = new_buffer_with_data(
        &heap_shared,
        bindings,
        MTLResourceOptions::StorageModeShared,
        "bindings",
    );

    // Build resource descriptor heap
    let acc_header_buffer_desc = MetalDescriptor::buffer(&acc_structure_gpu_header_buffer);
    let bindings_buffer_desc = MetalDescriptor::buffer(&bindings_buffer);
    let descriptor_count = vec![
        bindings_buffer_desc,   // SRV
        bindings_buffer_desc,   // UAV
        acc_header_buffer_desc, // SRV
        acc_header_buffer_desc, // UAV
    ];
    let resource_heap = new_buffer_with_data(
        &heap_shared,
        descriptor_count,
        MTLResourceOptions::StorageModeShared,
        "resource_heap",
    );

    // Build top level argument buffer
    let top_level_argument_buffer = [create_render_resource_handle(
        0,
        RenderResourceTag::Buffer,
        0,
    )];

    // get DXIL shaders
    let vs_dxil = include_bytes!("main.vs.dxil");
    let ps_dxil = include_bytes!("main.ps.dxil");
    let pipeline_state = create_vs_ps_pipeline_from_dxil(vs_dxil, ps_dxil, device);

    // Render loop
    let mut time_curr = Instant::now();
    let mut time_prev = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        autoreleasepool(|| {
            *control_flow = ControlFlow::Poll;
            // dbg!(&event);
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => (),
                },
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    // Calculate delta time
                    time_prev = time_curr;
                    time_curr = Instant::now();
                    let _delta_time = (time_curr - time_prev).as_secs_f32();

                    // Get next drawable
                    let drawable = layer.next_drawable().unwrap();

                    // Set up frame buffer
                    let render_pass_descriptor = RenderPassDescriptor::new();
                    let color_attachment = render_pass_descriptor
                        .color_attachments()
                        .object_at(0)
                        .expect("Failed to get color attachment");
                    color_attachment.set_texture(Some(drawable.texture()));
                    color_attachment.set_load_action(MTLLoadAction::Clear);
                    color_attachment.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.2, 1.0));
                    color_attachment.set_store_action(MTLStoreAction::Store);

                    // Draw
                    let command_buffer = command_queue.new_command_buffer();
                    let command_encoder =
                        command_buffer.new_render_command_encoder(render_pass_descriptor);

                    command_encoder
                        .use_heaps(&[&heap_private, &heap_shared], MTLRenderStages::all());
                    command_encoder.set_vertex_buffer(
                        BIND_POINT_DESCRIPTOR_HEAP,
                        Some(&resource_heap),
                        0,
                    );
                    command_encoder.set_fragment_buffer(
                        BIND_POINT_DESCRIPTOR_HEAP,
                        Some(&resource_heap),
                        0,
                    );
                    command_encoder.set_vertex_bytes(
                        BIND_POINT_ARGUMENT_BUFFER,
                        std::mem::size_of_val(&top_level_argument_buffer) as u64,
                        top_level_argument_buffer.as_ptr() as _,
                    );
                    command_encoder.set_fragment_bytes(
                        BIND_POINT_ARGUMENT_BUFFER,
                        std::mem::size_of_val(&top_level_argument_buffer) as u64,
                        top_level_argument_buffer.as_ptr() as _,
                    );
                    command_encoder.set_render_pipeline_state(&pipeline_state);
                    draw_primitives(command_encoder, MTLPrimitiveType::Triangle, 0, 6, 1, 0);
                    command_encoder.end_encoding();

                    // Present framebuffer
                    command_buffer.present_drawable(drawable);
                    command_buffer.commit();
                    command_buffer.wait_until_completed();
                }
                _ => {}
            }
        });
    });
}

fn create_vs_ps_pipeline_from_dxil(
    vs_dxil: &[u8],
    ps_dxil: &[u8],
    device: Device,
) -> metal::RenderPipelineState {
    let metal_irconverter = unsafe { libloading::Library::new("libmetalirconverter.dylib") }
        .expect("Failed to load metalirconverter!");
    let vs_mtl = compile_dxil_to_metallib(
        &metal_irconverter,
        vs_dxil,
        "main",
        IRShaderStage::IRShaderStageVertex,
    )
    .unwrap()
    .binary;
    let ps_mtl = compile_dxil_to_metallib(
        &metal_irconverter,
        ps_dxil,
        "main",
        IRShaderStage::IRShaderStageFragment,
    )
    .unwrap()
    .binary;

    let vs_library = device.new_library_with_data(&vs_mtl).unwrap();
    let ps_library = device.new_library_with_data(&ps_mtl).unwrap();

    let vert_fun = vs_library.get_function("main", None).unwrap();
    let frag_fun = ps_library.get_function("main", None).unwrap();

    // Create render pipeline
    let pipeline_desc = metal::RenderPipelineDescriptor::new();
    pipeline_desc.set_vertex_function(Some(&vert_fun));
    pipeline_desc.set_fragment_function(Some(&frag_fun));
    pipeline_desc.set_label("Render pipeline");
    let attachment = pipeline_desc.color_attachments().object_at(0).unwrap();
    attachment.set_pixel_format(MTLPixelFormat::RGBA16Float);
    attachment.set_blending_enabled(false);
    let pipeline_state = device.new_render_pipeline_state(&pipeline_desc).unwrap();
    pipeline_state
}

fn new_buffer_with_data<T>(
    heap: &metal::Heap,
    data: Vec<T>,
    options: MTLResourceOptions,
    name: &str,
) -> metal::Buffer {
    let vertex_buffer = heap
        .new_buffer((data.len() * size_of::<T>()) as _, options)
        .unwrap();
    vertex_buffer.set_label(name);
    let vertex_buffer_data = vertex_buffer.contents() as *mut T;
    unsafe { std::ptr::copy(data.as_ptr(), vertex_buffer_data, data.len()) }
    vertex_buffer
}

fn build_acceleration_structure(
    heap_shared: &metal::Heap,
    heap_private: &metal::Heap,
    vertex_buffer: metal::Buffer,
    index_buffer: &metal::Buffer,
    device: &Device,
    command_queue: &metal::CommandQueue,
) -> (metal::AccelerationStructure, metal::AccelerationStructure) {
    // Create geometry
    let geo_desc = metal::AccelerationStructureTriangleGeometryDescriptor::descriptor();
    geo_desc.set_vertex_format(metal::MTLAttributeFormat::Float3);
    geo_desc.set_vertex_buffer(Some(&vertex_buffer));
    geo_desc.set_vertex_buffer_offset(0);
    geo_desc.set_vertex_stride(size_of::<[f32; 3]>() as _);
    geo_desc.set_index_type(metal::MTLIndexType::UInt32);
    geo_desc.set_index_buffer(Some(index_buffer));
    geo_desc.set_index_buffer_offset(0);
    geo_desc.set_triangle_count(1);
    geo_desc.set_intersection_function_table_offset(0);
    geo_desc.set_opaque(true);

    // Create Blas
    let blas_desc = metal::PrimitiveAccelerationStructureDescriptor::descriptor();
    blas_desc.set_geometry_descriptors(metal::Array::from_owned_slice(&[geo_desc.into()]));
    let build_sizes = device.acceleration_structure_sizes_with_descriptor(&blas_desc);
    let scratch_buffer = heap_shared
        .new_buffer(
            build_sizes.build_scratch_buffer_size,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let cmd = command_queue.new_command_buffer();
    let enc = cmd.new_acceleration_structure_command_encoder();
    let blas = heap_private
        .new_acceleration_structure_with_size(build_sizes.acceleration_structure_size)
        .unwrap();
    enc.build_acceleration_structure(&blas, &blas_desc, &scratch_buffer, 0);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Create instance buffer
    let mut instance = metal::MTLIndirectAccelerationStructureInstanceDescriptor::default();
    instance.transformation_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ];
    instance.acceleration_structure_id = blas.gpu_resource_id()._impl;
    instance.mask = 0xFFFFFFFF;
    instance.options = metal::MTLAccelerationStructureInstanceOptions::Opaque;
    instance.user_id = 0;
    instance.intersection_function_table_offset = 0;
    let instance_buffer = new_buffer_with_data(
        heap_shared,
        vec![instance],
        MTLResourceOptions::StorageModeShared,
        "instance buffer",
    );

    // Create Tlas
    let tlas_desc = metal::InstanceAccelerationStructureDescriptor::descriptor();
    tlas_desc.set_instance_descriptor_buffer(&instance_buffer);
    tlas_desc.set_instance_descriptor_type(
        metal::MTLAccelerationStructureInstanceDescriptorType::Indirect,
    );
    tlas_desc.set_instance_count(1);

    let cmd = command_queue.new_command_buffer();
    let enc = cmd.new_acceleration_structure_command_encoder();
    let build_sizes = device.acceleration_structure_sizes_with_descriptor(&tlas_desc);
    let scratch_buffer = heap_shared
        .new_buffer(
            build_sizes.build_scratch_buffer_size,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let tlas = heap_private
        .new_acceleration_structure_with_size(build_sizes.acceleration_structure_size)
        .unwrap();

    enc.build_acceleration_structure(&tlas, &tlas_desc, &scratch_buffer, 0);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    (blas, tlas)
}

struct CompiledMetalShader {
    binary: Vec<u8>,
}

fn compile_dxil_to_metallib(
    metal_irconverter: &libloading::Library,
    dxil_binary: &[u8],
    entry_point: &str,
    shader_type: IRShaderStage,
) -> Result<CompiledMetalShader, Box<dyn std::error::Error>> {
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

    Ok(CompiledMetalShader {
        binary: mtl_binary.get_byte_code(),
    })
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

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MetalDescriptor {
    gpu_virtual_address: u64,
    texture_view_id: u64,
    metadata: u64,
}

impl MetalDescriptor {
    fn buffer(buffer: &metal::Buffer) -> Self {
        let buf_size_mask = 0xffffffff;
        let buf_size_offset = 0;
        let typed_buffer_offset = 63;

        let metadata =
            ((buffer.length() & buf_size_mask) << buf_size_offset) | (1 << typed_buffer_offset);

        MetalDescriptor {
            gpu_virtual_address: buffer.gpu_address(),
            texture_view_id: 0,
            metadata,
        }
    }
}

// Based on ir_raytracing.h:121 IRRaytracingAccelerationStructureGPUHeader struct
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RaytracingAccelerationStructureGPUHeader {
    pub acceleration_structure_id: u64,
    pub address_of_instance_contributions: u64,
    pub pad0: [u64; 4],
    pub pad1: [u32; 3],
    pub pad2: u32,
}

#[repr(C)]
struct DrawArgument {
    vertex_count_per_instance: u32,
    instance_count: u32,
    start_vertex_location: u32,
    start_instance_location: u32,
}

#[repr(C)]
struct DrawInfo {
    index_type: u32,
    primitive_topology: u32,
    max_input_primitives_per_mesh_threadgroup: u32,
    object_threadgroup_vertex_stride: u32,
    gs_instance_count: u32,
}

// Based on IRRuntimeDrawPrimitives function in metal_irconverter_runtime.h
fn draw_primitives(
    encoder: &metal::RenderCommandEncoderRef,
    primitive_type: MTLPrimitiveType,
    vertex_start: u64,
    vertex_count: u64,
    instance_count: u64,
    base_instance: u64,
) {
    let draw_params = DrawArgument {
        vertex_count_per_instance: vertex_count as u32,
        instance_count: instance_count as u32,
        start_vertex_location: 0,
        start_instance_location: 0,
    };
    let draw_info = DrawInfo {
        index_type: 0, // unused
        primitive_topology: primitive_type as u32,
        max_input_primitives_per_mesh_threadgroup: 0,
        object_threadgroup_vertex_stride: 0,
        gs_instance_count: 0,
    };
    encoder.set_vertex_bytes(
        BIND_POINT_ARGUMENT_BUFFER_DRAW_ARGUMENTS,
        size_of::<DrawArgument>() as _,
        &draw_params as *const DrawArgument as *const c_void,
    );
    encoder.set_vertex_bytes(
        BIND_POINT_ARGUMENT_BUFFER_UNIFORMS,
        size_of::<DrawInfo>() as _,
        &draw_info as *const DrawInfo as *const c_void,
    );
    encoder.draw_primitives_instanced_base_instance(
        primitive_type,
        vertex_start,
        vertex_count,
        instance_count,
        base_instance,
    );
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderResourceTag {
    Tlas,
    Buffer,
    Texture,
}

impl TryFrom<u32> for RenderResourceTag {
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Tlas),
            1 => Ok(Self::Buffer),
            2 => Ok(Self::Texture),
            _ => Err(()),
        }
    }

    type Error = ();
}

fn create_render_resource_handle(version: u8, tag: RenderResourceTag, index: u32) -> u32 {
    let version = version as u32;
    let tag = tag as u32;
    let access_type = 0;

    assert!(version < 64); // version wraps around, it's just to make sure invalid resources don't get another version
    assert!((tag & !0x3) == 0);
    assert!(index < (1 << 23));

    version << 26 | access_type << 25 | tag << 23 | index
}
