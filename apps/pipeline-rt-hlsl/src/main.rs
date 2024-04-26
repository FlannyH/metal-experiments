use std::ffi::{c_void, CString};
use std::mem::{self, size_of};
use std::sync::Arc;
use std::time::Instant;

use cocoa::appkit::NSView;
use core_graphics_types::geometry::CGSize;
use metal::foreign_types::ForeignType;
use metal::{
    Device, MTLClearColor, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderStages, MTLResourceOptions, MTLResourceUsage, MTLSize, MTLStoreAction,
    MTLTextureUsage, MetalLayer, RenderPassDescriptor, TextureDescriptor,
};
use objc::rc::autoreleasepool;
use objc::runtime::YES;
use saxaboom::{
    IRComparisonFunction, IRCompiler, IRFilter, IRHitGroupType, IRMetalLibBinary, IRObject,
    IRRootConstants, IRRootParameter1, IRRootParameter1_u, IRRootParameterType, IRRootSignature,
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

pub const INDIRECT_TRIANGLE_INTERSECTION_FUNCTION_NAME: &str =
    "irconverter.wrapper.intersection.function.triangle";
pub const INDIRECT_PROCEDURAL_INTERSECTION_FUNCTION_NAME: &str =
    "irconverter.wrapper.intersection.function.procedural";
pub const RAY_DISPATCH_INDIRECTION_KERNEL_NAME: &str = "RaygenIndirection";

const SBT_TRIANGLE_INTERSECTION_FUNCTION_INDEX: u64 = 0;
const SBT_BOX_INTERSECTION_FUNCTION_INDEX: u64 = 1;
const MAX_INTERSECTION_FUNCTIONS: u64 = 2;

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
    let device = Arc::new(Device::system_default().expect("Could not create device."));

    // Create metal layer
    let layer = MetalLayer::new();
    layer.set_device(&device);
    layer.set_pixel_format(MTLPixelFormat::RGBA16Float);
    layer.set_presents_with_transaction(false);
    layer.set_framebuffer_only(false);
    layer.set_drawable_size(CGSize {
        width: 1280.0,
        height: 720.0,
    });

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
        heap_desc_shared.set_hazard_tracking_mode(metal::MTLHazardTrackingMode::Tracked);
        device.new_heap(&heap_desc_shared)
    };
    let heap_private = {
        let heap_desc_private = metal::HeapDescriptor::new();
        heap_desc_private.set_size(128 * 1024 * 1024);
        heap_desc_private.set_storage_mode(metal::MTLStorageMode::Private);
        heap_desc_private.set_hazard_tracking_mode(metal::MTLHazardTrackingMode::Tracked);
        device.new_heap(&heap_desc_private)
    };

    // Commands
    let command_queue = device.new_command_queue();

    // Create vertex buffer
    let positions = vec![
        [100.0f32, 400.0f32, 3.0f32],
        [200.0f32, 100.0f32, 3.0f32],
        [300.0f32, 400.0f32, 3.0f32],
    ];
    let options = MTLResourceOptions::StorageModeShared;
    let position_buffer = new_buffer_with_data(&heap_shared, positions, options, "vertex buffer");

    // Create index buffer
    let indices = vec![0u32, 1, 2, 3, 4, 5];
    let index_buffer = new_buffer_with_data(&heap_shared, indices, options, "index buffer");

    let (blas, tlas) = build_acceleration_structure(
        &heap_shared,
        &heap_private,
        position_buffer,
        &index_buffer,
        &device,
        &command_queue,
    );

    drop(index_buffer);

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
    let bindings = vec![
        create_render_resource_handle(0, RenderResourceTag::Tlas, 2),
        create_render_resource_handle(0, RenderResourceTag::Texture, 4),
    ];
    let bindings_buffer = new_buffer_with_data(
        &heap_shared,
        bindings,
        MTLResourceOptions::StorageModeShared,
        "bindings",
    );

    // Make offscreen render target
    let offscreen_target_desc = metal::TextureDescriptor::new();
    offscreen_target_desc.set_width(1280);
    offscreen_target_desc.set_height(720);
    offscreen_target_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
    offscreen_target_desc.set_texture_type(metal::MTLTextureType::D2Array);
    offscreen_target_desc.set_storage_mode(metal::MTLStorageMode::Private);
    offscreen_target_desc.set_usage(
        MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget,
    );
    let offscreen_target = heap_private.new_texture(&offscreen_target_desc).unwrap();

    // Build resource descriptor heap
    let bindings_buffer_desc = MetalDescriptor::buffer(&bindings_buffer);
    let acc_header_buffer_desc = MetalDescriptor::buffer(&acc_structure_gpu_header_buffer);
    let offscreen_target_texture_desc = MetalDescriptor::texture(&offscreen_target);
    let descriptor_count = vec![
        bindings_buffer_desc,          // SRV
        bindings_buffer_desc,          // UAV
        acc_header_buffer_desc,        // SRV
        acc_header_buffer_desc,        // UAV
        offscreen_target_texture_desc, // SRV
        offscreen_target_texture_desc, // UAV
    ];
    let resource_heap = new_buffer_with_data(
        &heap_shared,
        descriptor_count,
        MTLResourceOptions::StorageModeShared,
        "resource_heap",
    );

    // Build top level argument buffer
    let top_level_argument_buffer = [
        // Bindings struct
        create_render_resource_handle(0, RenderResourceTag::Buffer, 0),
    ];

    // get DXIL shaders
    let rgen = include_bytes!("example.rgen.dxil");
    let rchit = include_bytes!("example.rchit.dxil");
    let rmiss = include_bytes!("example.rmiss.dxil");
    let metal_irconverter = unsafe { libloading::Library::new("libmetalirconverter.dylib") }
        .expect("Failed to get metalirconverter library");
    let compiled_rt = compile_rt_stages(
        &metal_irconverter,
        rgen,
        "main",
        rchit,
        "main",
        rmiss,
        "main",
    );

    // Compile pipeline
    let (visible_function_table, intersection_function_table, pipeline_state) = create_rt_pipeline(
        &metal_irconverter,
        &compiled_rt.rgen.binary,
        "rgen_main",
        &compiled_rt.chit.binary,
        "chit_main",
        &compiled_rt.miss.binary,
        "miss_main",
        &device,
    );

    // Create shader binding table
    let sbt_buffer = create_shader_binding_table(&heap_shared, 0, 1, 2);

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

                    // Set up environment
                    let command_buffer = command_queue.new_command_buffer();
                    let command_encoder: &metal::ComputeCommandEncoderRef =
                        command_buffer.new_compute_command_encoder();

                    let top_level_ab = new_buffer_with_data(
                        &heap_shared,
                        top_level_argument_buffer.to_vec(),
                        MTLResourceOptions::StorageModeShared,
                        "ray tracing top level argument buffer",
                    );

                    // Set dispatch info and SBT info
                    let sbt_start_address = sbt_buffer.buffer.gpu_address();
                    let mut sbt_strides = sbt_buffer.stride_data.clone();

                    sbt_strides.ray_generation_shader_record_start_address += sbt_start_address;
                    sbt_strides.hit_group_table_start_address += sbt_start_address;
                    sbt_strides.miss_shader_table_start_address += sbt_start_address;
                    sbt_strides.callable_shader_table_start_address += sbt_start_address;

                    let ray_gen_shader_record = VirtualAddressRange {
                        start_address: sbt_strides.ray_generation_shader_record_start_address,
                        size_in_bytes: sbt_strides.ray_generation_shader_record_size_in_bytes,
                    };

                    let hitg_shader_table = VirtualAddressRangeAndStride {
                        start_address: sbt_strides.hit_group_table_start_address,
                        size_in_bytes: sbt_strides.hit_group_table_size_in_bytes,
                        stride_in_bytes: sbt_strides.hit_group_table_stride_in_bytes,
                    };

                    let callable_shader_table = VirtualAddressRangeAndStride {
                        start_address: sbt_start_address,
                        size_in_bytes: 0u64,
                        stride_in_bytes: 0u64,
                    };

                    let miss_shader_table = VirtualAddressRangeAndStride {
                        start_address: sbt_strides.miss_shader_table_start_address,
                        size_in_bytes: sbt_strides.miss_shader_table_size_in_bytes,
                        stride_in_bytes: sbt_strides.miss_shader_table_stride_in_bytes,
                    };

                    let dispatch_rays_desc: DispatchRaysDescriptor = DispatchRaysDescriptor {
                        ray_gen_shader_record,
                        miss_shader_table,
                        hitg_shader_table,
                        callable_shader_table,
                        width: drawable.texture().width() as _,
                        height: drawable.texture().height() as _,
                        depth: 1,
                    };

                    let dispatch_ray_arguments = DispatchRaysArgument {
                        dispatch_rays_desc,
                        top_level_global_ab_gpu_address: top_level_ab.gpu_address(),
                        res_desc_heap_ab_gpu_address: resource_heap.gpu_address(),
                        smp_desc_heap_ab_gpu_address: 0,
                        visible_function_table_res_id: visible_function_table
                            .gpu_resource_id()
                            ._impl,
                        intersection_function_table_res_id: intersection_function_table
                            .gpu_resource_id()
                            ._impl,
                        pad: [0; 7],
                    };

                    let thread_group_size = MTLSize {
                        width: pipeline_state.max_total_threads_per_threadgroup(),
                        height: 1,
                        depth: 1,
                    };

                    let grid_size = MTLSize {
                        width: drawable.texture().width() as _,
                        height: drawable.texture().height() as _,
                        depth: 1,
                    };

                    command_encoder.set_compute_pipeline_state(&pipeline_state);
                    command_encoder.use_resources(
                        &[
                            &top_level_ab,
                            &visible_function_table,
                            &intersection_function_table,
                        ],
                        MTLResourceUsage::Read,
                    );
                    command_encoder.use_heaps(&[&heap_private, &heap_shared]);
                    // dbg!(&dispatch_ray_arguments);
                    command_encoder.set_bytes(
                        BIND_POINT_RAY_DISPATCH_ARGUMENTS,
                        size_of::<DispatchRaysArgument>() as u64,
                        &dispatch_ray_arguments as *const DispatchRaysArgument as *const c_void,
                    );
                    command_encoder.dispatch_threads(grid_size, thread_group_size);

                    let fence = device.new_fence();
                    command_encoder.update_fence(&fence);
                    command_encoder.end_encoding();

                    // Blit to drawable
                    let command_encoder = command_buffer.new_blit_command_encoder();
                    command_encoder.wait_for_fence(&fence);
                    command_encoder.copy_from_texture(
                        &offscreen_target,
                        0,
                        0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                        MTLSize {
                            width: offscreen_target.width(),
                            height: offscreen_target.height(),
                            depth: offscreen_target.depth(),
                        },
                        drawable.texture(),
                        0,
                        0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                    );
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

struct ShaderBindingTable {
    buffer: metal::Buffer,
    stride_data: SbtStrideData,
}

fn create_shader_binding_table(
    heap_shared: &metal::Heap,
    rgen_table_index: u64,
    chit_table_index: u64,
    miss_table_index: u64,
) -> ShaderBindingTable {
    // Create vector to hold all the shader identifiers
    let table_len = 3;

    let mut sbt_data: Vec<ShaderIdentifier> = vec![ShaderIdentifier::default(); table_len];

    // Shader handles
    sbt_data[rgen_table_index as usize].shader_handle = (rgen_table_index + 1) as u64;
    sbt_data[chit_table_index as usize].shader_handle = (chit_table_index + 1) as u64;
    sbt_data[miss_table_index as usize].shader_handle = (miss_table_index + 1) as u64;

    // Create the buffer
    dbg!(&sbt_data);
    let buffer = new_buffer_with_data(
        heap_shared,
        sbt_data,
        MTLResourceOptions::StorageModeShared,
        "shader binding table",
    );

    // Create the stride data
    const RECORD_STRIDE: u64 = size_of::<ShaderIdentifier>() as u64;
    let rgen_table_offset = rgen_table_index * RECORD_STRIDE;
    let miss_table_offset = miss_table_index * RECORD_STRIDE;
    let hitg_table_offset = chit_table_index * RECORD_STRIDE;
    let rgen_table_size = RECORD_STRIDE;
    let miss_table_size = 1 * RECORD_STRIDE;
    let hitg_table_size = 1 * RECORD_STRIDE;

    let stride_data = SbtStrideData {
        ray_generation_shader_record_start_address: rgen_table_offset as u64,
        ray_generation_shader_record_size_in_bytes: rgen_table_size as u64,
        miss_shader_table_start_address: miss_table_offset as u64,
        miss_shader_table_size_in_bytes: miss_table_size as u64,
        miss_shader_table_stride_in_bytes: 0u64,
        hit_group_table_start_address: hitg_table_offset as u64,
        hit_group_table_size_in_bytes: hitg_table_size as u64,
        hit_group_table_stride_in_bytes: 0u64,
        callable_shader_table_start_address: 0u64,
        callable_shader_table_size_in_bytes: 0u64,
        callable_shader_table_stride_in_bytes: 0u64,
    };
    // dbg!(&stride_data);

    ShaderBindingTable {
        buffer,
        stride_data,
    }
}

fn create_rt_pipeline(
    metal_irconverter: &libloading::Library,
    rgen_mtl: &[u8],
    rgen_entry: &str,
    chit_mtl: &[u8],
    chit_entry: &str,
    miss_mtl: &[u8],
    miss_entry: &str,
    device: &Device,
) -> (
    metal::VisibleFunctionTable,
    metal::IntersectionFunctionTable,
    metal::ComputePipelineState,
) {
    // Synthesize indirect triangle intersection shader
    let indirect_triangle_intersection_bytecode = synthesize_indirect_intersection_function(
        metal_irconverter,
        saxaboom::IRHitGroupType::IRHitGroupTypeTriangles,
    );
    let indirect_triangle_intersection_library = device
        .new_library_with_data(&indirect_triangle_intersection_bytecode)
        .unwrap();
    let indirect_triangle_intersection_function = indirect_triangle_intersection_library
        .get_function(INDIRECT_TRIANGLE_INTERSECTION_FUNCTION_NAME, None)
        .unwrap();

    // Synthesize indirect box intersection function
    let indirect_box_intersection_bytecode = synthesize_indirect_intersection_function(
        metal_irconverter,
        saxaboom::IRHitGroupType::IRHitGroupTypeProceduralPrimitive,
    );
    let indirect_box_intersection_library = device
        .new_library_with_data(&indirect_box_intersection_bytecode)
        .unwrap();
    let indirect_box_intersection_function = indirect_box_intersection_library
        .get_function(INDIRECT_PROCEDURAL_INTERSECTION_FUNCTION_NAME, None)
        .unwrap();

    // Synthesize a ray dispatch compute function
    let ray_dispatch_bytecode = synthesize_indirect_ray_dispatch_function(metal_irconverter);
    let ray_dispatch_library = device
        .new_library_with_data(&ray_dispatch_bytecode)
        .unwrap();
    let ray_dispatch_function = ray_dispatch_library
        .get_function(RAY_DISPATCH_INDIRECTION_KERNEL_NAME, None)
        .unwrap();

    // Create metal functions
    let rgen_lib = device.new_library_with_data(rgen_mtl).unwrap();
    let chit_lib = device.new_library_with_data(chit_mtl).unwrap();
    let miss_lib = device.new_library_with_data(miss_mtl).unwrap();
    let rgen_fun = rgen_lib.get_function(rgen_entry, None).unwrap();
    let chit_fun = chit_lib.get_function(chit_entry, None).unwrap();
    let miss_fun = miss_lib.get_function(miss_entry, None).unwrap();

    // Linked functions
    let linked_function_refs: &[&metal::FunctionRef] = &[
        &rgen_fun,
        &chit_fun,
        &miss_fun,
        &indirect_triangle_intersection_function,
        &indirect_box_intersection_function,
    ];
    let linked_functions = metal::LinkedFunctions::new();
    linked_functions.set_functions(linked_function_refs);

    // Create compute pipeline
    let compute_desc = metal::ComputePipelineDescriptor::new();
    compute_desc.set_compute_function(Some(&ray_dispatch_function));
    compute_desc.set_linked_functions(&linked_functions);
    let compute_pipeline = device.new_compute_pipeline_state(&compute_desc).unwrap();

    // Create visible function table
    let vft_desc = metal::VisibleFunctionTableDescriptor::new();
    vft_desc.set_function_count(4); // rgen, chit, miss, + 1 extra because index 0 is the null function index
    let vft = compute_pipeline.new_visible_function_table_with_descriptor(&vft_desc);

    // Load functions into visible function table
    vft.set_function(
        compute_pipeline
            .function_handle_with_function(&rgen_fun)
            .unwrap(),
        1,
    );
    vft.set_function(
        compute_pipeline
            .function_handle_with_function(&chit_fun)
            .unwrap(),
        2,
    );
    vft.set_function(
        compute_pipeline
            .function_handle_with_function(&miss_fun)
            .unwrap(),
        3,
    );

    // Create intersection function
    let ift_desc = metal::IntersectionFunctionTableDescriptor::new();
    ift_desc.set_function_count(MAX_INTERSECTION_FUNCTIONS);

    let handle_triangle = compute_pipeline
        .function_handle_with_function(&indirect_triangle_intersection_function)
        .unwrap();
    let handle_box = compute_pipeline
        .function_handle_with_function(&indirect_box_intersection_function)
        .unwrap();

    let ift = compute_pipeline.new_intersection_function_table_with_descriptor(&ift_desc);
    ift.set_function(handle_triangle, SBT_TRIANGLE_INTERSECTION_FUNCTION_INDEX);
    ift.set_function(handle_box, SBT_BOX_INTERSECTION_FUNCTION_INDEX);

    // todo(lily): maybe store the ift and vft somewhere?
    (vft, ift, compute_pipeline)
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
    let mut instance = metal::MTLAccelerationStructureInstanceDescriptor::default();
    instance.transformation_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ];
    // instance.acceleration_structure_id = blas.gpu_resource_id()._impl;
    instance.acceleration_structure_index = 0;
    instance.mask = 0xFFFFFFFF;
    instance.options = metal::MTLAccelerationStructureInstanceOptions::Opaque;
    // instance.user_id = 0;
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
    // tlas_desc.set_instance_descriptor_buffer_offset(0);
    tlas_desc.set_instance_descriptor_type(
        metal::MTLAccelerationStructureInstanceDescriptorType::Default,
    );
    tlas_desc.set_instance_count(1);
    let instances = metal::Array::from_slice(&[blas.as_ref()]);
    tlas_desc.set_instanced_acceleration_structures(instances);

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

struct MetalRayTracingCompileInfo<'a> {
    closest_hit_mask: u64,
    any_hit_mask: u64,
    miss_mask: u64,
    max_recursion_depth: i32,
    hit_group_type: IRHitGroupType,
    entry_point_remap: &'a String,
}

fn compile_dxil_to_metallib(
    metal_irconverter: &libloading::Library,
    obj: &IRObject,
    entry_point: &str,
    shader_type: IRShaderStage,
    rt_params: &MetalRayTracingCompileInfo,
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
                    num32_bit_values: 2,
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
    let mut c = IRCompiler::new(lib)?;
    c.set_global_root_signature(&root_sig);

    // Set RT params
    c.set_ray_tracing_pipeline_arguments(
        32, // D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES
        saxaboom::IRRaytracingPipelineFlags::IRRaytracingPipelineFlagNone,
        rt_params.closest_hit_mask,
        rt_params.miss_mask,
        rt_params.any_hit_mask,
        u64::MAX, // All
        rt_params.max_recursion_depth,
    );
    c.set_hitgroup_type(rt_params.hit_group_type);
    c.set_entry_point_name(
        CString::new(rt_params.entry_point_remap.as_str())
            .unwrap()
            .as_c_str(),
    );

    let entry_point_cstring = CString::new(entry_point).unwrap();

    let mut mtl_binary = IRMetalLibBinary::new(lib)?;
    let mtllib = c.alloc_compile_and_link(&entry_point_cstring, &obj)?;
    mtllib.get_metal_lib_binary(shader_type, &mut mtl_binary);

    let mut mtl_reflection = IRShaderReflection::new(lib)?;
    mtllib.get_reflection(shader_type, &mut mtl_reflection);

    dbg!(mtl_reflection.needs_function_constants());

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
    fn texture(argument: &metal::Texture) -> Self {
        MetalDescriptor {
            gpu_virtual_address: 0,
            texture_view_id: argument.gpu_resource_id()._impl,
            metadata: 0,
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

struct CompiledRaytracingMetal {
    rgen: CompiledMetalShader,
    chit: CompiledMetalShader,
    miss: CompiledMetalShader,
}

fn compile_rt_stages(
    metal_irconverter: &libloading::Library,
    rgen_dxil: &[u8],
    rgen_entry: &str,
    chit_dxil: &[u8],
    chit_entry: &str,
    miss_dxil: &[u8],
    miss_entry: &str,
) -> CompiledRaytracingMetal {
    // Convert DXIL to IRObject
    let rgen = IRObject::create_from_dxil(metal_irconverter, rgen_dxil).unwrap();
    let chit = IRObject::create_from_dxil(metal_irconverter, chit_dxil).unwrap();
    let miss = IRObject::create_from_dxil(metal_irconverter, miss_dxil).unwrap();

    // Gather intrinsic masks
    let chit_entry_cstring = CString::new(chit_entry).unwrap();
    let miss_entry_cstring = CString::new(miss_entry).unwrap();
    let closest_hit_mask = chit.gather_raytracing_intrinsics(&chit_entry_cstring);
    let miss_mask = miss.gather_raytracing_intrinsics(&miss_entry_cstring);
    let any_hit_mask = 0;

    // Get names for remapped entry points
    let remap_rgen = format!("rgen_{rgen_entry}");
    let remap_chit = format!("chit_{rgen_entry}");
    let remap_miss = format!("miss_{rgen_entry}");

    // Compile to metallib
    let rgen_metallib = compile_dxil_to_metallib(
        metal_irconverter,
        &rgen,
        rgen_entry,
        IRShaderStage::IRShaderStageRayGeneration,
        &MetalRayTracingCompileInfo {
            closest_hit_mask,
            any_hit_mask,
            miss_mask,
            max_recursion_depth: 1,
            hit_group_type: IRHitGroupType::IRHitGroupTypeTriangles,
            entry_point_remap: &remap_rgen,
        },
    )
    .unwrap();
    let chit_metallib = compile_dxil_to_metallib(
        metal_irconverter,
        &chit,
        chit_entry,
        IRShaderStage::IRShaderStageClosestHit,
        &MetalRayTracingCompileInfo {
            closest_hit_mask,
            any_hit_mask,
            miss_mask,
            max_recursion_depth: 1,
            hit_group_type: IRHitGroupType::IRHitGroupTypeTriangles,
            entry_point_remap: &remap_chit,
        },
    )
    .unwrap();
    let miss_metallib = compile_dxil_to_metallib(
        metal_irconverter,
        &miss,
        miss_entry,
        IRShaderStage::IRShaderStageMiss,
        &MetalRayTracingCompileInfo {
            closest_hit_mask,
            any_hit_mask,
            miss_mask,
            max_recursion_depth: 1,
            hit_group_type: IRHitGroupType::IRHitGroupTypeTriangles,
            entry_point_remap: &remap_miss,
        },
    )
    .unwrap();
    CompiledRaytracingMetal {
        rgen: rgen_metallib,
        chit: chit_metallib,
        miss: miss_metallib,
    }
}
pub fn synthesize_indirect_intersection_function(
    metal_irconverter: &libloading::Library,
    intersection_type: IRHitGroupType,
) -> Vec<u8> {
    let mut compiler = IRCompiler::new(metal_irconverter).unwrap();
    let mut target_metallib = IRMetalLibBinary::new(metal_irconverter).unwrap();
    compiler.set_hitgroup_type(intersection_type);
    compiler.synthesize_indirect_intersection_function(&mut target_metallib);
    target_metallib.get_byte_code()
}

pub fn synthesize_indirect_ray_dispatch_function(
    metal_irconverter: &libloading::Library,
) -> Vec<u8> {
    let mut compiler = IRCompiler::new(metal_irconverter).unwrap();
    let mut target_metallib = IRMetalLibBinary::new(metal_irconverter).unwrap();
    compiler.synthesize_indirect_ray_dispatch_function(&mut target_metallib);
    target_metallib.get_byte_code()
}
// Based on the IRShaderIdentifier struct in Metal Shader Converter's ir_raytracing.h header
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::NoUninit)]
struct ShaderIdentifier {
    intersection_shader_handle: u64,
    shader_handle: u64,
    local_root_signature_samplers_buffer: u64,
    pad0: u64,
}
#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct SbtStrideData {
    pub ray_generation_shader_record_start_address: u64,
    pub ray_generation_shader_record_size_in_bytes: u64,

    pub miss_shader_table_start_address: u64,
    pub miss_shader_table_size_in_bytes: u64,
    pub miss_shader_table_stride_in_bytes: u64,

    pub hit_group_table_start_address: u64,
    pub hit_group_table_size_in_bytes: u64,
    pub hit_group_table_stride_in_bytes: u64,

    pub callable_shader_table_start_address: u64,
    pub callable_shader_table_size_in_bytes: u64,
    pub callable_shader_table_stride_in_bytes: u64,
}
#[repr(C)]
#[derive(Debug)]
struct VirtualAddressRange {
    start_address: u64,
    size_in_bytes: u64,
}

#[repr(C)]
#[derive(Debug)]
struct VirtualAddressRangeAndStride {
    start_address: u64,
    size_in_bytes: u64,
    stride_in_bytes: u64,
}

#[repr(C)]
#[derive(Debug)]
struct DispatchRaysDescriptor {
    ray_gen_shader_record: VirtualAddressRange,
    miss_shader_table: VirtualAddressRangeAndStride,
    hitg_shader_table: VirtualAddressRangeAndStride,
    callable_shader_table: VirtualAddressRangeAndStride,
    width: u32,
    height: u32,
    depth: u32,
}

#[repr(C)]
#[derive(Debug)]
struct DispatchRaysArgument {
    dispatch_rays_desc: DispatchRaysDescriptor,
    top_level_global_ab_gpu_address: u64, // same as bound to BIND_POINT_ARGUMENT_BUFFER
    res_desc_heap_ab_gpu_address: u64,    // same as bound to BIND_POINT_DESCRIPTOR_HEAP
    smp_desc_heap_ab_gpu_address: u64,
    visible_function_table_res_id: u64,
    intersection_function_table_res_id: u64,
    pad: [u32; 7],
}
