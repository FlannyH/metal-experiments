use std::ffi::c_void;
use std::mem::{self, size_of};
use std::time::Instant;

use cocoa::appkit::NSView;
use metal::foreign_types::ForeignType;
use metal::{
    CompileOptions, Device, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLRenderStages,
    MTLResourceOptions, MTLResourceUsage, MTLStoreAction, MetalLayer, RenderPassDescriptor,
};
use objc::rc::autoreleasepool;
use objc::runtime::YES;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::platform::macos::WindowExtMacOS;

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

    // Create view - a sort of canvas where you draw graphics using Metal commands
    unsafe {
        let view = window.ns_view() as cocoa::base::id;
        view.setWantsLayer(YES);
        view.setLayer(mem::transmute(layer.as_ptr()));
    }

    // Commands
    let command_queue = device.new_command_queue();

    // Create vertex buffer
    let positions = vec![
        [100.0f32, 100.1f32, 0.0f32],
        [200.0f32, 100.2f32, 0.0f32],
        [302.0f32, 403.0f32, 0.0f32],
    ];
    let vertex_buffer = device.new_buffer_with_data(
        positions.as_ptr() as *const c_void,
        (positions.len() * size_of::<[f32; 3]>()) as _,
        MTLResourceOptions::StorageModeShared,
    );

    // Create index buffer
    let indices = vec![0u32, 1, 2, 3, 4, 5];
    let index_buffer = device.new_buffer_with_data(
        indices.as_ptr() as *const c_void,
        (indices.len() * size_of::<u32>()) as _,
        MTLResourceOptions::StorageModeShared,
    );

    // Create geometry
    let geo_desc = metal::AccelerationStructureTriangleGeometryDescriptor::descriptor();
    geo_desc.set_vertex_format(metal::MTLAttributeFormat::Float3);
    geo_desc.set_vertex_buffer(Some(&vertex_buffer));
    geo_desc.set_vertex_buffer_offset(0);
    geo_desc.set_vertex_stride(size_of::<[f32; 3]>() as _);
    geo_desc.set_index_type(metal::MTLIndexType::UInt32);
    geo_desc.set_index_buffer(Some(&index_buffer));
    geo_desc.set_index_buffer_offset(0);
    geo_desc.set_triangle_count(1);
    geo_desc.set_intersection_function_table_offset(0);
    geo_desc.set_opaque(true);

    // Create Blas
    let blas_desc = metal::PrimitiveAccelerationStructureDescriptor::descriptor();
    blas_desc.set_geometry_descriptors(metal::Array::from_owned_slice(&[geo_desc.into()]));
    let build_sizes = device.acceleration_structure_sizes_with_descriptor(&blas_desc);
    let scratch_buffer = device.new_buffer(
        build_sizes.build_scratch_buffer_size,
        MTLResourceOptions::StorageModeShared,
    );
    let cmd = command_queue.new_command_buffer();
    let enc = cmd.new_acceleration_structure_command_encoder();
    let blas = device.new_acceleration_structure_with_size(build_sizes.acceleration_structure_size);
    enc.build_acceleration_structure(&blas, &blas_desc, &scratch_buffer, 0);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Create instance buffer
    let mut instances = metal::MTLIndirectAccelerationStructureInstanceDescriptor::default();
    instances.transformation_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ];
    instances.acceleration_structure_id = blas.gpu_resource_id()._impl;
    instances.mask = 0xFFFFFFFF;
    instances.options = metal::MTLAccelerationStructureInstanceOptions::Opaque;
    instances.user_id = 0;
    instances.intersection_function_table_offset = 0;
    let instance_buffer = device.new_buffer_with_data(
        <*const metal::MTLIndirectAccelerationStructureInstanceDescriptor>::cast(&instances),
        size_of::<metal::MTLIndirectAccelerationStructureInstanceDescriptor>() as _,
        MTLResourceOptions::StorageModeShared,
    );

    // Create Tlas
    let tlas_desc = metal::InstanceAccelerationStructureDescriptor::descriptor();
    tlas_desc.set_instance_descriptor_buffer(&instance_buffer);
    tlas_desc.set_instance_descriptor_buffer_offset(0);
    tlas_desc.set_instance_descriptor_stride(72);
    tlas_desc.set_instance_descriptor_type(
        metal::MTLAccelerationStructureInstanceDescriptorType::Indirect,
    );
    tlas_desc.set_instance_count(1);

    let cmd = command_queue.new_command_buffer();
    let enc = cmd.new_acceleration_structure_command_encoder();
    let build_sizes = device.acceleration_structure_sizes_with_descriptor(&tlas_desc);
    let scratch_buffer = device.new_buffer(
        build_sizes.build_scratch_buffer_size,
        MTLResourceOptions::StorageModeShared,
    );
    let tlas = device.new_acceleration_structure_with_size(build_sizes.acceleration_structure_size);

    enc.build_acceleration_structure(&tlas, &tlas_desc, &scratch_buffer, 0);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Vertex shader
    let shaders = include_str!("shaders.metal");

    let compile_options = CompileOptions::new();
    let shader_lib = device
        .new_library_with_source(shaders, &compile_options)
        .unwrap();
    let vert_fun = shader_lib.get_function("v_main", None).unwrap();
    let frag_fun = shader_lib.get_function("f_main", None).unwrap();

    // Create full screen quad
    let quad_positions = vec![
        [-1.0f32, -1.0f32],
        [-1.0f32, 1.0f32],
        [1.0f32, 1.0f32],
        [-1.0f32, -1.0f32],
        [1.0f32, 1.0f32],
        [1.0f32, -1.0f32],
    ];
    let quad_buffer = device.new_buffer_with_data(
        quad_positions.as_ptr() as *const c_void,
        (quad_positions.len() * size_of::<f32>() * 2) as _,
        MTLResourceOptions::StorageModeShared,
    );

    // Create render pipeline
    let pipeline_desc = metal::RenderPipelineDescriptor::new();
    pipeline_desc.set_vertex_function(Some(&vert_fun));
    pipeline_desc.set_fragment_function(Some(&frag_fun));
    pipeline_desc.set_label("Render pipeline");
    let attachment = pipeline_desc.color_attachments().object_at(0).unwrap();
    attachment.set_pixel_format(MTLPixelFormat::RGBA16Float);
    attachment.set_blending_enabled(false);
    let pipeline_state = device.new_render_pipeline_state(&pipeline_desc).unwrap();

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

                    // Render
                    let command_buffer = command_queue.new_command_buffer();
                    let command_encoder =
                        command_buffer.new_render_command_encoder(render_pass_descriptor);
                    command_encoder.use_resource_at(
                        &tlas,
                        MTLResourceUsage::Read,
                        MTLRenderStages::all(),
                    );
                    command_encoder.use_resource_at(
                        &blas,
                        MTLResourceUsage::Read,
                        MTLRenderStages::all(),
                    );
                    command_encoder.set_render_pipeline_state(&pipeline_state);
                    command_encoder.set_vertex_buffer(0, Some(&quad_buffer), 0);
                    command_encoder.set_fragment_acceleration_structure(0, Some(&tlas));
                    command_encoder.draw_indexed_primitives(
                        metal::MTLPrimitiveType::Triangle,
                        6,
                        metal::MTLIndexType::UInt32,
                        &index_buffer,
                        0,
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
