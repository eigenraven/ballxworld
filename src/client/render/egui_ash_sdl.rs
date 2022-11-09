//! Original license (kept for this file): Apache-2.0 and MIT (c) Orito Itsuki
//!
//! Original: This is the [egui](https://github.com/emilk/egui) integration crate for
//! [winit](https://github.com/rust-windowing/winit), [ash](https://github.com/MaikKlein/ash)
//! and [vk_mem](https://github.com/gwihlidal/vk-mem-rs).
//!
//! # Example
//! ```sh
//! cargo run --example example
//! ```
//!
//! # Usage
//!
//! ```
//! fn main() -> Result<()> {
//!     let event_loop = EventLoop::new();
//!     // (1) Call Integration::new() in App::new().
//!     let mut app = App::new(&event_loop)?;
//!
//!     event_loop.run(move |event, _, control_flow| {
//!         *control_flow = ControlFlow::Poll;
//!         // (2) Call integration.handle_event(&event).
//!         app.egui_integration.handle_event(&event);
//!         match event {
//!             Event::WindowEvent {
//!                 event: WindowEvent::CloseRequested,
//!                 ..
//!             } => *control_flow = ControlFlow::Exit,
//!             Event::WindowEvent {
//!                 event: WindowEvent::Resized(_),
//!                 ..
//!             } => {
//!                 // (3) Call integration.recreate_swapchain(...) in app.recreate_swapchain().
//!                 app.recreate_swapchain().unwrap();
//!             }
//!             Event::MainEventsCleared => app.window.request_redraw(),
//!             Event::RedrawRequested(_window_id) => {
//!                 // (4) Call integration.begin_frame(), integration.end_frame(&mut window),
//!                 // integration.context().tessellate(shapes), integration.paint(...)
//!                 // in app.draw().
//!                 app.draw().unwrap();
//!             }
//!             _ => (),
//!         }
//!     })
//! }
//! // (5) Call integration.destroy() when drop app.
//! ```
//!
//! [Full example is in examples directory](https://github.com/MatchaChoco010/egui_winit_ash_vk_mem/tree/main/examples/example)
#![warn(missing_docs)]

use crate::{client::render::vulkan::allocation_cbs, vk};
use bxw_util::bytemuck::bytes_of;
use bxw_util::fnv::FnvHashMap;
use egui::{
    emath::{pos2, vec2},
    epaint::Primitive,
    Context, Key,
};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use std::ffi::CStr;
use vk_mem_3_erupt as vma;

use crate::client::render::vkhelpers::{OwnedDescriptorSet, OwnedImage};
use crate::client::render::vulkan::INFLIGHT_FRAMES;
use crate::client::render::RenderingContext;

use super::{
    vkhelpers::VulkanDeviceObject, vulkan::RenderingHandles, InPassFrameContext,
    PrePassFrameContext,
};

/// egui integration with winit, ash and vma.
pub struct EguiIntegration {
    logical_width: u32,
    logical_height: u32,
    scale_factor: f64,
    context: Context,
    raw_input: egui::RawInput,
    mouse_pos: egui::Pos2,
    current_cursor_icon: egui::CursorIcon,

    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    vertex_buffers: Vec<vk::Buffer>,
    vertex_buffer_allocations: Vec<vma::Allocation>,
    index_buffers: Vec<vk::Buffer>,
    index_buffer_allocations: Vec<vma::Allocation>,
    managed_images: FnvHashMap<u64, (OwnedImage, OwnedDescriptorSet)>,
    clipped_primitives: Vec<egui::ClippedPrimitive>,
}
impl EguiIntegration {
    /// Create an instance of the integration.
    pub fn new(rctx: &mut RenderingContext) -> Self {
        let font_definitions = egui::FontDefinitions::default();
        let style = egui::Style::default();
        let device = &rctx.handles.device;
        let (logical_width, logical_height) = rctx.window.vulkan_drawable_size();
        let scale_factor: f64 = logical_width as f64 / (rctx.window.size().0 as f64).max(1.0);

        // Create context
        let context = Context::default();
        context.set_fonts(font_definitions);
        context.set_style(style);
        context.set_visuals(egui::Visuals::dark());

        // Create raw_input
        let raw_input = egui::RawInput {
            pixels_per_point: Some(scale_factor as f32),
            screen_rect: Some(egui::Rect::from_min_size(
                Default::default(),
                vec2(logical_width as f32, logical_height as f32),
            )),
            time: Some(0.0),
            ..Default::default()
        };

        // Create mouse pos and modifier state (These values are overwritten by handle events)
        let mouse_pos = pos2(0.0, 0.0);

        // Create DescriptorPool
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfoBuilder::new()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(1024)
                    .pool_sizes(&[vk::DescriptorPoolSizeBuilder::new()
                        ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1024)]),
                allocation_cbs(),
            )
        }
        .expect("Failed to create descriptor pool.");

        // Create DescriptorSetLayouts
        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&[
                    vk::DescriptorSetLayoutBindingBuilder::new()
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .binding(0)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                ]),
                allocation_cbs(),
            )
        }
        .expect("Failed to create descriptor set layout.");

        // Create PipelineLayout
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfoBuilder::new()
                    .set_layouts(&[descriptor_set_layout])
                    .push_constant_ranges(&[
                        vk::PushConstantRangeBuilder::new()
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .offset(0)
                            .size(std::mem::size_of::<f32>() as u32 * 2), // screen size
                    ]),
                allocation_cbs(),
            )
        }
        .expect("Failed to create pipeline layout.");

        // Create Pipeline
        let pipeline = {
            let bindings = [vk::VertexInputBindingDescriptionBuilder::new()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(
                    4 * std::mem::size_of::<f32>() as u32 + 4 * std::mem::size_of::<u8>() as u32,
                )];

            let attributes = [
                // position
                vk::VertexInputAttributeDescriptionBuilder::new()
                    .binding(0)
                    .offset(0)
                    .location(0)
                    .format(vk::Format::R32G32_SFLOAT),
                // uv
                vk::VertexInputAttributeDescriptionBuilder::new()
                    .binding(0)
                    .offset(8)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT),
                // color
                vk::VertexInputAttributeDescriptionBuilder::new()
                    .binding(0)
                    .offset(16)
                    .location(2)
                    .format(vk::Format::R8G8B8A8_UNORM),
            ];

            let vertex_shader_module = rctx
                .handles
                .load_shader_module("res/shaders/egui.vert.spv")
                .expect("Couldn't load vertex egui shader");
            let fragment_shader_module = rctx
                .handles
                .load_shader_module("res/shaders/egui.frag.spv")
                .expect("Couldn't load fragment egui shader");
            let main_function_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
            let pipeline_shader_stages = [
                vk::PipelineShaderStageCreateInfoBuilder::new()
                    .stage(vk::ShaderStageFlagBits::VERTEX)
                    .module(vertex_shader_module)
                    .name(main_function_name),
                vk::PipelineShaderStageCreateInfoBuilder::new()
                    .stage(vk::ShaderStageFlagBits::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(main_function_name),
            ];

            let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_info = vk::PipelineViewportStateCreateInfoBuilder::new()
                .viewport_count(1)
                .scissor_count(1);
            let rasterization_info = vk::PipelineRasterizationStateCreateInfoBuilder::new()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0);
            let stencil_op = vk::StencilOpStateBuilder::new()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
                .depth_test_enable(false)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::ALWAYS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(stencil_op)
                .back(stencil_op);
            let color_blend_attachments = [vk::PipelineColorBlendAttachmentStateBuilder::new()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)];
            let color_blend_info = vk::PipelineColorBlendStateCreateInfoBuilder::new()
                .attachments(&color_blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfoBuilder::new().dynamic_states(&dynamic_states);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfoBuilder::new()
                .vertex_attribute_descriptions(&attributes)
                .vertex_binding_descriptions(&bindings);
            let multisample_info = vk::PipelineMultisampleStateCreateInfoBuilder::new()
                .rasterization_samples(rctx.handles.sample_count);

            let pipeline_create_info = [vk::GraphicsPipelineCreateInfoBuilder::new()
                .stages(&pipeline_shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_info)
                .viewport_state(&viewport_info)
                .rasterization_state(&rasterization_info)
                .multisample_state(&multisample_info)
                .depth_stencil_state(&depth_stencil_info)
                .color_blend_state(&color_blend_info)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .render_pass(rctx.handles.mainpass)
                .subpass(0)];

            let pipeline = unsafe {
                device.create_graphics_pipelines(
                    rctx.pipeline_cache,
                    &pipeline_create_info,
                    allocation_cbs(),
                )
            }
            .expect("Failed to create egui graphics pipeline.")[0];
            unsafe {
                device.destroy_shader_module(vertex_shader_module, allocation_cbs());
                device.destroy_shader_module(fragment_shader_module, allocation_cbs());
            }
            pipeline
        };

        // Create Sampler
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfoBuilder::new()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(false)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                allocation_cbs(),
            )
        }
        .expect("Failed to create sampler.");

        // Create vertex buffer and index buffer
        let mut vertex_buffers = vec![];
        let mut vertex_buffer_allocations = vec![];
        let mut index_buffers = vec![];
        let mut index_buffer_allocations = vec![];
        let vmalloc = rctx.handles.vmalloc.lock();
        for _ in 0..INFLIGHT_FRAMES {
            let (vertex_buffer, vertex_buffer_allocation, _info) = vmalloc
                .create_buffer(
                    &vk::BufferCreateInfoBuilder::new()
                        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .size(Self::vertex_buffer_size()),
                    &vma::AllocationCreateInfo {
                        usage: vma::MemoryUsage::CpuToGpu,
                        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                        ..Default::default()
                    },
                )
                .expect("Failed to create vertex buffer.");
            let (index_buffer, index_buffer_allocation, _info) = vmalloc
                .create_buffer(
                    &vk::BufferCreateInfoBuilder::new()
                        .usage(vk::BufferUsageFlags::INDEX_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .size(Self::index_buffer_size()),
                    &vma::AllocationCreateInfo {
                        usage: vma::MemoryUsage::CpuToGpu,
                        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                        ..Default::default()
                    },
                )
                .expect("Failed to create index buffer.");
            vertex_buffers.push(vertex_buffer);
            vertex_buffer_allocations.push(vertex_buffer_allocation);
            index_buffers.push(index_buffer);
            index_buffer_allocations.push(index_buffer_allocation);
        }
        drop(vmalloc);

        Self {
            logical_width,
            logical_height,
            scale_factor,
            context,
            raw_input,
            mouse_pos,
            current_cursor_icon: egui::CursorIcon::None,

            descriptor_pool,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            sampler,
            vertex_buffers,
            vertex_buffer_allocations,
            index_buffers,
            index_buffer_allocations,
            managed_images: Default::default(),
            clipped_primitives: Default::default(),
        }
    }

    // vertex buffer size
    fn vertex_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    // index buffer size
    fn index_buffer_size() -> u64 {
        1024 * 1024 * 2
    }

    /// handling winit event.
    pub fn handle_event(&mut self, winit_event: &Event, rctx: &mut RenderingContext) {
        match winit_event {
            Event::Window {
                timestamp: _timestamp,
                window_id: _window_id,
                win_event,
            } => match win_event {
                // window size changed
                WindowEvent::SizeChanged(..) => {
                    let (logical_width, logical_height) = rctx.window.vulkan_drawable_size();
                    self.logical_width = logical_width;
                    self.logical_height = logical_height;
                    let scale_factor: f64 =
                        logical_width as f64 / (rctx.window.size().0 as f64).max(1.0);
                    self.raw_input.pixels_per_point = Some(scale_factor as f32);
                    self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
                        Default::default(),
                        vec2(logical_width as f32, logical_height as f32),
                    ));
                }
                // mouse out
                WindowEvent::Leave => {
                    self.raw_input.events.push(egui::Event::PointerGone);
                }
                _ => (),
            },
            // mouse click
            Event::MouseButtonDown {
                mouse_btn, x, y, ..
            } => {
                if let Some(button) = Self::sdl_to_egui_mouse_button(*mouse_btn) {
                    self.raw_input.events.push(egui::Event::PointerButton {
                        pos: egui::pos2(*x as f32, *y as f32),
                        button,
                        pressed: true,
                        modifiers: Self::sdl_to_egui_modifiers(
                            rctx.window.subsystem().sdl().keyboard().mod_state(),
                        ),
                    });
                }
            }
            Event::MouseButtonUp {
                mouse_btn, x, y, ..
            } => {
                if let Some(button) = Self::sdl_to_egui_mouse_button(*mouse_btn) {
                    self.raw_input.events.push(egui::Event::PointerButton {
                        pos: egui::pos2(*x as f32, *y as f32),
                        button,
                        pressed: false,
                        modifiers: Self::sdl_to_egui_modifiers(
                            rctx.window.subsystem().sdl().keyboard().mod_state(),
                        ),
                    });
                }
            }
            // mouse wheel
            Event::MouseWheel { x, y, .. } => {
                let wheel_factor = 1.0;
                self.raw_input.events.push(egui::Event::Scroll(
                    vec2(*x as f32, *y as f32) * wheel_factor,
                ));
            }
            // mouse move
            Event::MouseMotion { x, y, .. } => {
                let pixels_per_point = self
                    .raw_input
                    .pixels_per_point
                    .unwrap_or_else(|| self.context.pixels_per_point());
                let pos = pos2(*x as f32 / pixels_per_point, *y as f32 / pixels_per_point);
                self.raw_input.events.push(egui::Event::PointerMoved(pos));
                self.mouse_pos = pos;
            }
            // keyboard inputs
            Event::KeyDown {
                keycode, keymod, ..
            } => {
                let modifiers = Self::sdl_to_egui_modifiers(*keymod);
                if let Some(keycode) = keycode {
                    let keycode = *keycode;
                    let is_ctrl = modifiers.ctrl;
                    if is_ctrl && keycode == Keycode::C {
                        self.raw_input.events.push(egui::Event::Copy);
                    } else if is_ctrl && keycode == Keycode::X {
                        self.raw_input.events.push(egui::Event::Cut);
                    } else if is_ctrl && keycode == Keycode::V {
                        if let Ok(contents) = rctx.window.subsystem().clipboard().clipboard_text() {
                            self.raw_input.events.push(egui::Event::Text(contents));
                        }
                    } else if let Some(key) = Self::sdl_to_egui_key_code(keycode) {
                        self.raw_input.events.push(egui::Event::Key {
                            key,
                            pressed: true,
                            modifiers,
                        })
                    }
                }
            }
            Event::KeyUp {
                keycode, keymod, ..
            } => {
                let modifiers = Self::sdl_to_egui_modifiers(*keymod);
                if let Some(keycode) = keycode {
                    let keycode = *keycode;
                    if let Some(key) = Self::sdl_to_egui_key_code(keycode) {
                        self.raw_input.events.push(egui::Event::Key {
                            key,
                            pressed: false,
                            modifiers,
                        })
                    }
                }
            }
            // receive character
            Event::TextInput { text, .. } => {
                // remove control character
                if text.chars().all(|c| c.is_ascii_control()) {
                    return;
                }
                self.raw_input.events.push(egui::Event::Text(text.clone()));
            }
            _ => (),
        }
    }

    fn sdl_to_egui_key_code(key: Keycode) -> Option<egui::Key> {
        Some(match key {
            Keycode::Down => Key::ArrowDown,
            Keycode::Left => Key::ArrowLeft,
            Keycode::Right => Key::ArrowRight,
            Keycode::Up => Key::ArrowUp,
            Keycode::Escape => Key::Escape,
            Keycode::Tab => Key::Tab,
            Keycode::Backspace => Key::Backspace,
            Keycode::Return => Key::Enter,
            Keycode::Space => Key::Space,
            Keycode::Insert => Key::Insert,
            Keycode::Delete => Key::Delete,
            Keycode::Home => Key::Home,
            Keycode::End => Key::End,
            Keycode::PageUp => Key::PageUp,
            Keycode::PageDown => Key::PageDown,
            Keycode::Num0 => Key::Num0,
            Keycode::Num1 => Key::Num1,
            Keycode::Num2 => Key::Num2,
            Keycode::Num3 => Key::Num3,
            Keycode::Num4 => Key::Num4,
            Keycode::Num5 => Key::Num5,
            Keycode::Num6 => Key::Num6,
            Keycode::Num7 => Key::Num7,
            Keycode::Num8 => Key::Num8,
            Keycode::Num9 => Key::Num9,
            Keycode::A => Key::A,
            Keycode::B => Key::B,
            Keycode::C => Key::C,
            Keycode::D => Key::D,
            Keycode::E => Key::E,
            Keycode::F => Key::F,
            Keycode::G => Key::G,
            Keycode::H => Key::H,
            Keycode::I => Key::I,
            Keycode::J => Key::J,
            Keycode::K => Key::K,
            Keycode::L => Key::L,
            Keycode::M => Key::M,
            Keycode::N => Key::N,
            Keycode::O => Key::O,
            Keycode::P => Key::P,
            Keycode::Q => Key::Q,
            Keycode::R => Key::R,
            Keycode::S => Key::S,
            Keycode::T => Key::T,
            Keycode::U => Key::U,
            Keycode::V => Key::V,
            Keycode::W => Key::W,
            Keycode::X => Key::X,
            Keycode::Y => Key::Y,
            Keycode::Z => Key::Z,
            Keycode::F1 => Key::F1,
            Keycode::F2 => Key::F2,
            Keycode::F3 => Key::F3,
            Keycode::F4 => Key::F4,
            Keycode::F5 => Key::F5,
            Keycode::F6 => Key::F6,
            Keycode::F7 => Key::F7,
            Keycode::F8 => Key::F8,
            Keycode::F9 => Key::F9,
            Keycode::F10 => Key::F10,
            Keycode::F11 => Key::F11,
            Keycode::F12 => Key::F12,
            Keycode::F13 => Key::F13,
            Keycode::F14 => Key::F14,
            Keycode::F15 => Key::F15,
            _ => return None,
        })
    }

    fn sdl_to_egui_modifiers(modifiers: sdl2::keyboard::Mod) -> egui::Modifiers {
        use sdl2::keyboard::Mod;
        egui::Modifiers {
            alt: modifiers.intersects(Mod::LALTMOD | Mod::RALTMOD),
            ctrl: modifiers.intersects(Mod::LCTRLMOD | Mod::RCTRLMOD),
            shift: modifiers.intersects(Mod::LSHIFTMOD | Mod::RSHIFTMOD),
            mac_cmd: false,
            command: modifiers.intersects(Mod::LCTRLMOD | Mod::RCTRLMOD),
        }
    }

    fn sdl_to_egui_mouse_button(button: sdl2::mouse::MouseButton) -> Option<egui::PointerButton> {
        use sdl2::mouse::MouseButton;
        Some(match button {
            MouseButton::Left => egui::PointerButton::Primary,
            MouseButton::Right => egui::PointerButton::Secondary,
            MouseButton::Middle => egui::PointerButton::Middle,
            _ => return None,
        })
    }

    /// Convert from [`egui::CursorIcon`] to [`winit::window::CursorIcon`].
    fn egui_to_sdl_cursor_icon(cursor_icon: egui::CursorIcon) -> Option<sdl2::mouse::SystemCursor> {
        Some(match cursor_icon {
            egui::CursorIcon::Default => sdl2::mouse::SystemCursor::Arrow,
            egui::CursorIcon::PointingHand => sdl2::mouse::SystemCursor::Hand,
            egui::CursorIcon::ResizeHorizontal => sdl2::mouse::SystemCursor::SizeWE,
            egui::CursorIcon::ResizeNeSw => sdl2::mouse::SystemCursor::SizeNESW,
            egui::CursorIcon::ResizeNwSe => sdl2::mouse::SystemCursor::SizeNWSE,
            egui::CursorIcon::ResizeVertical => sdl2::mouse::SystemCursor::SizeNS,
            egui::CursorIcon::Text => sdl2::mouse::SystemCursor::IBeam,
            egui::CursorIcon::Grab => sdl2::mouse::SystemCursor::Hand,
            egui::CursorIcon::Grabbing => sdl2::mouse::SystemCursor::Hand,
            egui::CursorIcon::None => return None,
            egui::CursorIcon::ContextMenu => sdl2::mouse::SystemCursor::Arrow,
            egui::CursorIcon::Help => sdl2::mouse::SystemCursor::Arrow,
            egui::CursorIcon::Progress => sdl2::mouse::SystemCursor::WaitArrow,
            egui::CursorIcon::Wait => sdl2::mouse::SystemCursor::Wait,
            egui::CursorIcon::Cell => sdl2::mouse::SystemCursor::Crosshair,
            egui::CursorIcon::Crosshair => sdl2::mouse::SystemCursor::Crosshair,
            egui::CursorIcon::VerticalText => sdl2::mouse::SystemCursor::IBeam,
            egui::CursorIcon::Alias => sdl2::mouse::SystemCursor::Hand,
            egui::CursorIcon::Copy => sdl2::mouse::SystemCursor::Arrow,
            egui::CursorIcon::Move => sdl2::mouse::SystemCursor::SizeAll,
            egui::CursorIcon::NoDrop => sdl2::mouse::SystemCursor::No,
            egui::CursorIcon::NotAllowed => sdl2::mouse::SystemCursor::No,
            egui::CursorIcon::AllScroll => sdl2::mouse::SystemCursor::Arrow,
            egui::CursorIcon::ZoomIn => sdl2::mouse::SystemCursor::SizeAll,
            egui::CursorIcon::ZoomOut => sdl2::mouse::SystemCursor::SizeAll,
            egui::CursorIcon::ResizeEast => sdl2::mouse::SystemCursor::SizeNESW,
            egui::CursorIcon::ResizeSouthEast => sdl2::mouse::SystemCursor::SizeNESW,
            egui::CursorIcon::ResizeSouth => sdl2::mouse::SystemCursor::SizeNESW,
            egui::CursorIcon::ResizeSouthWest => sdl2::mouse::SystemCursor::SizeNESW,
            egui::CursorIcon::ResizeWest => sdl2::mouse::SystemCursor::SizeNESW,
            egui::CursorIcon::ResizeNorthWest => sdl2::mouse::SystemCursor::SizeNESW,
            egui::CursorIcon::ResizeNorth => sdl2::mouse::SystemCursor::SizeNESW,
            egui::CursorIcon::ResizeNorthEast => sdl2::mouse::SystemCursor::SizeNESW,
            egui::CursorIcon::ResizeColumn => sdl2::mouse::SystemCursor::SizeWE,
            egui::CursorIcon::ResizeRow => sdl2::mouse::SystemCursor::SizeNS,
        })
    }

    /// Get [`egui::CtxRef`].
    pub fn context(&self) -> Context {
        self.context.clone()
    }

    /// Update textures etc.
    pub fn prepass_draw(
        &mut self,
        command_buffer: vk::CommandBuffer,
        fctx: &mut PrePassFrameContext,
        run_ui: impl FnOnce(&Context),
    ) {
        self.logical_width = fctx.dims[0];
        self.logical_height = fctx.dims[1];
        self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
            Default::default(),
            egui::vec2(self.logical_width as f32, self.logical_height as f32),
        ));
        let egui::FullOutput {
            platform_output: output,
            shapes: clipped_shapes,
            textures_delta,
            ..
        } = self.context.run(self.raw_input.clone(), run_ui);
        self.raw_input.time = Some(self.raw_input.time.unwrap_or(0.0) + fctx.delta_time);
        self.raw_input.events.clear();
        let sdl_video = fctx.rctx.window.subsystem();
        // handle links
        if let Some(egui::output::OpenUrl { url, .. }) = &output.open_url {
            if let Err(err) = sdl2::url::open_url(url) {
                eprintln!("Failed to open url: {}", err);
            }
        }

        // handle clipboard
        if !output.copied_text.is_empty() {
            if let Err(err) = sdl_video
                .clipboard()
                .set_clipboard_text(&output.copied_text)
            {
                bxw_util::log::error!("Copy/Cut error: {}", err);
            }
        }

        // handle cursor icon
        let mut _cursor = None;
        if self.current_cursor_icon != output.cursor_icon {
            _cursor = Self::egui_to_sdl_cursor_icon(output.cursor_icon);
            self.current_cursor_icon = output.cursor_icon;
        }

        // handle text input
        {
            let tinput = sdl_video.text_input();
            let active = tinput.is_active();
            if !active && output.mutable_text_under_cursor {
                tinput.start();
            } else if active && !output.mutable_text_under_cursor {
                tinput.stop();
            }
            if output.mutable_text_under_cursor {
                if let Some(cursor_pos) = output.text_cursor_pos {
                    tinput.set_rect(sdl2::rect::Rect::new(
                        cursor_pos.x as i32,
                        cursor_pos.y as i32,
                        16,
                        16,
                    ));
                }
            }
        }

        // update font texture
        self.update_textures(command_buffer, textures_delta, fctx.rctx);

        self.clipped_primitives = self.context.tessellate(clipped_shapes);
    }

    /// Record paint commands.
    pub fn inpass_draw(
        &mut self,
        command_buffer: vk::CommandBuffer,
        fctx: &mut InPassFrameContext,
    ) {
        // update time
        self.raw_input.time = Some(self.raw_input.time.unwrap_or(0.0) + fctx.delta_time);

        let device = &fctx.rctx.handles.device;

        // map buffers
        let vmalloc = fctx.rctx.handles.vmalloc.lock();
        let mut vertex_buffer_ptr = vmalloc
            .map_memory(&self.vertex_buffer_allocations[fctx.inflight_index])
            .expect("Failed to map buffers.");
        let vertex_buffer_ptr_end =
            unsafe { vertex_buffer_ptr.add(Self::vertex_buffer_size() as usize) };
        let mut index_buffer_ptr = vmalloc
            .map_memory(&self.index_buffer_allocations[fctx.inflight_index])
            .expect("Failed to map buffers.");
        let index_buffer_ptr_end =
            unsafe { index_buffer_ptr.add(Self::index_buffer_size() as usize) };
        drop(vmalloc);

        // bind resources
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.vertex_buffers[fctx.inflight_index]],
                &[0],
            );
            device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffers[fctx.inflight_index],
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::ViewportBuilder::new()
                    .x(0.0)
                    .y(0.0)
                    .width(self.logical_width as f32)
                    .height(self.logical_height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );
            let wsz = fctx.rctx.window.size();
            let (width_points, height_points) = (wsz.0 as f32, wsz.1 as f32);
            let width_bytes = bytes_of(&width_points);
            let height_bytes = bytes_of(&height_points);
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                width_bytes.len() as u32,
                width_bytes.as_ptr() as *const std::ffi::c_void,
            );
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                width_bytes.len() as u32,
                height_bytes.len() as u32,
                height_bytes.as_ptr() as *const std::ffi::c_void,
            );
        }

        // render meshes
        let mut vertex_base = 0;
        let mut index_base = 0;
        for egui::ClippedPrimitive {
            clip_rect: rect,
            primitive: mesh,
        } in &self.clipped_primitives
        {
            if let Primitive::Mesh(mesh) = mesh {
                // update texture
                let descriptor = match mesh.texture_id {
                    egui::TextureId::User(_id) => None, // TODO
                    egui::TextureId::Managed(id) => {
                        self.managed_images.get(&id).map(|(_img, ds)| ds.1)
                    }
                    _ => None,
                };
                if let Some(descriptor) = descriptor {
                    unsafe {
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            &[descriptor],
                            &[],
                        );
                    }
                }

                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    continue;
                }

                let v_slice = &mesh.vertices;
                let v_size = std::mem::size_of_val(&v_slice[0]);
                let v_copy_size = v_slice.len() * v_size;

                let i_slice = &mesh.indices;
                let i_size = std::mem::size_of_val(&i_slice[0]);
                let i_copy_size = i_slice.len() * i_size;

                let vertex_buffer_ptr_next = unsafe { vertex_buffer_ptr.add(v_copy_size) };
                let index_buffer_ptr_next = unsafe { index_buffer_ptr.add(i_copy_size) };

                if vertex_buffer_ptr_next >= vertex_buffer_ptr_end
                    || index_buffer_ptr_next >= index_buffer_ptr_end
                {
                    panic!("egui paint out of memory");
                }

                // map memory
                unsafe { vertex_buffer_ptr.copy_from(v_slice.as_ptr() as *const u8, v_copy_size) };
                unsafe { index_buffer_ptr.copy_from(i_slice.as_ptr() as *const u8, i_copy_size) };

                vertex_buffer_ptr = vertex_buffer_ptr_next;
                index_buffer_ptr = index_buffer_ptr_next;

                // record draw commands
                unsafe {
                    let min = rect.min;
                    let min = egui::Pos2 {
                        x: min.x * self.scale_factor as f32,
                        y: min.y * self.scale_factor as f32,
                    };
                    let min = egui::Pos2 {
                        x: f32::clamp(min.x, 0.0, self.logical_width as f32),
                        y: f32::clamp(min.y, 0.0, self.logical_height as f32),
                    };
                    let max = rect.max;
                    let max = egui::Pos2 {
                        x: max.x * self.scale_factor as f32,
                        y: max.y * self.scale_factor as f32,
                    };
                    let max = egui::Pos2 {
                        x: f32::clamp(max.x, min.x, self.logical_width as f32),
                        y: f32::clamp(max.y, min.y, self.logical_height as f32),
                    };
                    device.cmd_set_scissor(
                        command_buffer,
                        0,
                        &[vk::Rect2DBuilder::new()
                            .offset(
                                vk::Offset2DBuilder::new()
                                    .x(min.x.round() as i32)
                                    .y(min.y.round() as i32)
                                    .build(),
                            )
                            .extent(
                                vk::Extent2DBuilder::new()
                                    .width((max.x.round() - min.x) as u32)
                                    .height((max.y.round() - min.y) as u32)
                                    .build(),
                            )],
                    );
                    device.cmd_draw_indexed(
                        command_buffer,
                        mesh.indices.len() as u32,
                        1,
                        index_base,
                        vertex_base,
                        0,
                    );
                }

                vertex_base += mesh.vertices.len() as i32;
                index_base += mesh.indices.len() as u32;
            }
        }

        // unmap buffers
        let vmalloc = fctx.rctx.handles.vmalloc.lock();
        vmalloc.flush_allocation(
            &self.vertex_buffer_allocations[fctx.inflight_index],
            0,
            vk::WHOLE_SIZE as usize,
        );
        vmalloc.flush_allocation(
            &self.index_buffer_allocations[fctx.inflight_index],
            0,
            vk::WHOLE_SIZE as usize,
        );
        vmalloc.unmap_memory(&self.vertex_buffer_allocations[fctx.inflight_index]);
        vmalloc.unmap_memory(&self.index_buffer_allocations[fctx.inflight_index]);
    }

    fn update_textures(
        &mut self,
        command_buffer: vk::CommandBuffer,
        textures: egui::TexturesDelta,
        rctx: &mut RenderingContext,
    ) {
        let mut vmalloc = rctx.handles.vmalloc.lock();
        // Enqueue destruction of old images until after they're done being used
        for id in textures.free.iter() {
            if let egui::TextureId::Managed(id) = id {
                if let Some((img, ds)) = self.managed_images.remove(id) {
                    rctx.handles.enqueue_destroy(Box::new(ds));
                    rctx.handles.enqueue_destroy(Box::new(img));
                }
            }
        }
        for (id, data) in textures.set.iter() {
            if let egui::TextureId::Managed(id) = *id {
                let (img, ds) = self.managed_images.entry(id).or_default();
                if data.is_whole() || ds.1.is_null() || img.allocation.is_none() {
                    if !ds.1.is_null() {
                        rctx.handles.enqueue_destroy(Box::new(std::mem::take(ds)));
                    }
                    if img.allocation.is_some() {
                        rctx.handles.enqueue_destroy(Box::new(std::mem::take(img)));
                    }
                    *img = OwnedImage::from(
                        &mut vmalloc,
                        &rctx.handles,
                        &vk::ImageCreateInfoBuilder::new()
                            .format(vk::Format::R8G8B8A8_UNORM)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
                            .samples(vk::SampleCountFlagBits::_1)
                            .tiling(vk::ImageTiling::OPTIMAL)
                            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .image_type(vk::ImageType::_2D)
                            .mip_levels(1)
                            .array_layers(1)
                            .extent(vk::Extent3D {
                                width: data.image.width() as u32,
                                height: data.image.height() as u32,
                                depth: 1,
                            }),
                        &vma::AllocationCreateInfo {
                            usage: vma::MemoryUsage::GpuOnly,
                            ..Default::default()
                        },
                        vk::ImageViewType::_2D,
                        vk::ImageAspectFlags::COLOR,
                    );
                    img.give_name(&rctx.handles, || format!("egui-{}", id));
                    *ds = OwnedDescriptorSet::from(
                        &rctx.handles,
                        self.descriptor_pool,
                        self.descriptor_set_layout,
                    );
                    // TODO: linear/nearest sampler
                    unsafe {
                        rctx.handles.device.update_descriptor_sets(
                            &[vk::WriteDescriptorSetBuilder::new()
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .dst_set(ds.1)
                                .image_info(&[vk::DescriptorImageInfoBuilder::new()
                                    .image_view(img.image_view)
                                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                    .sampler(self.sampler)])
                                .dst_binding(0)],
                            &[],
                        );
                    }
                }
                let (size, bytes): ([usize; 2], Vec<u8>) = match &data.image {
                    egui::ImageData::Font(font) => (font.size, {
                        let pixels = font.srgba_pixels(None);
                        let mut v = Vec::with_capacity(pixels.len() * 4);
                        for px in pixels {
                            v.extend_from_slice(&px.to_array());
                        }
                        v
                    }),
                    egui::ImageData::Color(color) => (color.size, {
                        let pixels = &color.pixels;
                        let mut v = Vec::with_capacity(pixels.len() * 4);
                        for px in pixels {
                            v.extend_from_slice(&px.to_array());
                        }
                        v
                    }),
                };
                let offset = data.pos.unwrap_or([0, 0]);
                let copy_desc = vk::BufferImageCopyBuilder::new()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D {
                        x: offset[0] as i32,
                        y: offset[1] as i32,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: size[0] as u32,
                        height: size[1] as u32,
                        depth: 1,
                    });
                rctx.handles.enqueue_destroy(Box::new(img.upload_bytes(
                    &mut vmalloc,
                    &rctx.handles,
                    command_buffer,
                    copy_desc,
                    &bytes[..],
                )));
            }
        }
    }

    /// destroy vk objects.
    ///
    /// # Safety
    /// This method release vk objects memory that is not managed by Rust.
    pub fn destroy(&mut self, handles: &RenderingHandles) {
        let device = &handles.device;
        let mut vmalloc = handles.vmalloc.lock();
        for (_id, mut img) in self.managed_images.drain() {
            img.1.destroy(&mut vmalloc, handles);
            img.0.destroy(&mut vmalloc, handles);
        }
        for i in 0..self.index_buffers.len() {
            vmalloc.destroy_buffer(self.index_buffers[i], &self.index_buffer_allocations[i]);
        }
        for i in 0..self.vertex_buffers.len() {
            vmalloc.destroy_buffer(self.vertex_buffers[i], &self.vertex_buffer_allocations[i]);
        }
        drop(vmalloc);
        unsafe {
            device.destroy_sampler(self.sampler, allocation_cbs());
            device.destroy_pipeline(self.pipeline, allocation_cbs());
            device.destroy_pipeline_layout(self.pipeline_layout, allocation_cbs());
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, allocation_cbs());
            device.destroy_descriptor_pool(self.descriptor_pool, allocation_cbs());
        }
    }
}
