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
use egui::{
    emath::{pos2, vec2},
    epaint::ClippedShape,
    Context, Key,
};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use std::ffi::CStr;

use crate::client::render::vulkan::INFLIGHT_FRAMES;
use crate::client::render::RenderingContext;

use super::{vulkan::RenderingHandles, InPassFrameContext, PrePassFrameContext};

/// egui integration with winit, ash and vk_mem_erupt.
pub struct EguiIntegration {
    logical_width: u32,
    logical_height: u32,
    scale_factor: f64,
    context: Context,
    raw_input: egui::RawInput,
    mouse_pos: egui::Pos2,
    current_cursor_icon: egui::CursorIcon,

    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    vertex_buffers: Vec<vk::Buffer>,
    vertex_buffer_allocations: Vec<vk_mem_erupt::Allocation>,
    index_buffers: Vec<vk::Buffer>,
    index_buffer_allocations: Vec<vk_mem_erupt::Allocation>,
    font_image_staging_buffer: vk::Buffer,
    font_image_staging_buffer_allocation: vk_mem_erupt::Allocation,
    font_image: vk::Image,
    font_image_allocation: vk_mem_erupt::Allocation,
    font_image_view: vk::ImageView,
    font_image_size: (u64, u64),
    font_image_version: u64,
    font_descriptor_sets: vk::SmallVec<vk::DescriptorSet>,

    user_texture_layout: vk::DescriptorSetLayout,
    user_textures: Vec<Option<vk::DescriptorSet>>,
}
impl EguiIntegration {
    /// Create an instance of the integration.
    pub fn new(
        font_definitions: egui::FontDefinitions,
        style: egui::Style,
        rctx: &mut RenderingContext,
    ) -> Self {
        let device = &rctx.handles.device;
        let (logical_width, logical_height) = rctx.window.vulkan_drawable_size();
        let scale_factor: f64 = logical_width as f64 / (rctx.window.size().0 as f64).max(1.0);

        // Create context
        let context = Context::default();
        context.set_fonts(font_definitions);
        context.set_style(style);

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
                None,
            )
        }
        .expect("Failed to create descriptor pool.");

        // Create DescriptorSetLayouts
        let descriptor_set_layouts = {
            let mut sets = vec![];
            for _ in 0..INFLIGHT_FRAMES {
                sets.push(
                    unsafe {
                        device.create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&[
                                vk::DescriptorSetLayoutBindingBuilder::new()
                                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .descriptor_count(1)
                                    .binding(0)
                                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                            ]),
                            None,
                        )
                    }
                    .expect("Failed to create descriptor set layout."),
                );
            }
            sets
        };

        // Create PipelineLayout
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfoBuilder::new()
                    .set_layouts(&descriptor_set_layouts)
                    .push_constant_ranges(&[
                        vk::PushConstantRangeBuilder::new()
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .offset(0)
                            .size(std::mem::size_of::<f32>() as u32 * 2), // screen size
                    ]),
                None,
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
                None,
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
                    &vk_mem_erupt::AllocationCreateInfo {
                        usage: vk_mem_erupt::MemoryUsage::CpuToGpu,
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
                    &vk_mem_erupt::AllocationCreateInfo {
                        usage: vk_mem_erupt::MemoryUsage::CpuToGpu,
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

        // Create font image and anything related to it
        // These values will be uploaded at rendering time
        let font_image_staging_buffer = Default::default();
        let font_image_staging_buffer_allocation = vk_mem_erupt::Allocation::null();
        let font_image = Default::default();
        let font_image_allocation = vk_mem_erupt::Allocation::null();
        let font_image_view = Default::default();
        let font_image_size = (0, 0);
        let font_image_version = 0;
        let font_descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfoBuilder::new()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&descriptor_set_layouts),
            )
        }
        .expect("Failed to create descriptor sets.");

        // User Textures
        let user_texture_layout = unsafe {
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
        let user_textures = vec![];

        Self {
            logical_width,
            logical_height,
            scale_factor,
            context,
            raw_input,
            mouse_pos,
            current_cursor_icon: egui::CursorIcon::None,

            descriptor_pool,
            descriptor_set_layouts,
            pipeline_layout,
            pipeline,
            sampler,
            vertex_buffers,
            vertex_buffer_allocations,
            index_buffers,
            index_buffer_allocations,
            font_image_staging_buffer,
            font_image_staging_buffer_allocation,
            font_image,
            font_image_allocation,
            font_image_view,
            font_image_size,
            font_image_version,
            font_descriptor_sets,

            user_texture_layout,
            user_textures,
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
        })
    }

    /// begin frame.
    pub fn begin_frame(&mut self) {
        self.context.begin_frame(self.raw_input.take());
    }

    /// end frame.
    pub fn end_frame(
        &mut self,
        sdl_video: &sdl2::VideoSubsystem,
    ) -> (
        egui::PlatformOutput,
        Vec<ClippedShape>,
        Option<sdl2::mouse::SystemCursor>,
    ) {
        let egui::FullOutput {
            platform_output: output,
            shapes: clipped_shapes,
            textures_delta,
            ..
        } = self.context.end_frame();

        // handle links
        if let Some(egui::output::OpenUrl { url, .. }) = &output.open_url {
            let _ = url;
            // TODO(https://github.com/Rust-SDL2/rust-sdl2/issues/1119)
            /*if let Err(err) = webbrowser::open(url) {
                eprintln!("Failed to open url: {}", err);
            }*/
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
        let mut cursor = None;
        if self.current_cursor_icon != output.cursor_icon {
            cursor = Self::egui_to_sdl_cursor_icon(output.cursor_icon);
            self.current_cursor_icon = output.cursor_icon;
        }

        (output, clipped_shapes, cursor)
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
    ) {
        self.logical_width = fctx.dims[0];
        self.logical_height = fctx.dims[1];
        self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
            Default::default(),
            egui::vec2(self.logical_width as f32, self.logical_height as f32),
        ));
        // update font texture
        // FIXME: self.upload_font_texture(command_buffer, &self.context.fonts().texture(), fctx.rctx);
    }

    /// Record paint commands.
    pub fn paint(
        &mut self,
        command_buffer: vk::CommandBuffer,
        clipped_meshes: Vec<egui::ClippedMesh>,
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
        for egui::ClippedMesh(rect, mesh) in clipped_meshes {
            // update texture
            unsafe {
                if let egui::TextureId::User(id) = mesh.texture_id {
                    if let Some(descriptor_set) = self.user_textures[id as usize] {
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            &[descriptor_set],
                            &[],
                        );
                    } else {
                        eprintln!(
                            "This UserTexture has already been unregistered: {:?}",
                            mesh.texture_id
                        );
                        continue;
                    }
                } else {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[self.font_descriptor_sets[fctx.inflight_index]],
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

    fn upload_font_texture(
        &mut self,
        command_buffer: vk::CommandBuffer,
        textures: egui::TexturesDelta,
        rctx: &mut RenderingContext,
    ) {
        /*assert_eq!(texture.pixels.len(), texture.width * texture.height);
        let device = &rctx.handles.device;

        // check version
        if texture.version == self.font_image_version {
            return;
        }

        // FIXME: Use multiple images, gc the unused ones
        unsafe {
            device
                .device_wait_idle()
                .expect("Failed to wait device idle");
        }

        let dimensions = (texture.width as u64, texture.height as u64);
        let data = texture
            .pixels
            .iter()
            .flat_map(|&r| vec![r, r, r, r])
            .collect::<Vec<_>>();

        let vmalloc = rctx.handles.vmalloc.lock();
        // free prev staging buffer
        vmalloc.destroy_buffer(
            self.font_image_staging_buffer,
            &self.font_image_staging_buffer_allocation,
        );

        // free font image
        unsafe {
            device.destroy_image_view(self.font_image_view, None);
        }
        vmalloc.destroy_image(self.font_image, &self.font_image_allocation);

        // create font image
        let (font_image_staging_buffer, font_image_staging_buffer_allocation, _info) = vmalloc
            .create_buffer(
                &vk::BufferCreateInfoBuilder::new()
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .size(dimensions.0 * dimensions.1 * 4),
                &vk_mem_erupt::AllocationCreateInfo {
                    usage: vk_mem_erupt::MemoryUsage::CpuOnly,
                    required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                    ..Default::default()
                },
            )
            .expect("Couldn't create font image staging buffer");
        self.font_image_staging_buffer = font_image_staging_buffer;
        self.font_image_staging_buffer_allocation = font_image_staging_buffer_allocation;
        let (font_image, font_image_allocation, _info) = vmalloc
            .create_image(
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
                        width: dimensions.0 as u32,
                        height: dimensions.1 as u32,
                        depth: 1,
                    }),
                &vk_mem_erupt::AllocationCreateInfo {
                    usage: vk_mem_erupt::MemoryUsage::GpuOnly,
                    ..Default::default()
                },
            )
            .expect("Failed to create image.");
        self.font_image = font_image;
        self.font_image_allocation = font_image_allocation;
        self.font_image_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfoBuilder::new()
                    .image(self.font_image)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .view_type(vk::ImageViewType::_2D)
                    .subresource_range(
                        vk::ImageSubresourceRangeBuilder::new()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .base_mip_level(0)
                            .layer_count(1)
                            .level_count(1)
                            .build(),
                    ),
                None,
            )
        }
        .expect("Failed to create image view.");
        self.font_image_size = dimensions;
        self.font_image_version = texture.version;

        // update descriptor set
        for &font_descriptor_set in self.font_descriptor_sets.iter() {
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSetBuilder::new()
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_set(font_descriptor_set)
                        .image_info(&[vk::DescriptorImageInfoBuilder::new()
                            .image_view(self.font_image_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .sampler(self.sampler)])
                        .dst_binding(0)],
                    &[],
                );
            }
        }

        // map memory
        let ptr = vmalloc
            .map_memory(&self.font_image_staging_buffer_allocation)
            .expect("Failed to map memory");
        unsafe {
            ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        vmalloc.unmap_memory(&self.font_image_staging_buffer_allocation);
        drop(vmalloc);
        // record buffer staging commands to command buffer
        unsafe {
            // update image layout to transfer dst optimal
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrierBuilder::new()
                    .image(self.font_image)
                    .subresource_range(
                        vk::ImageSubresourceRangeBuilder::new()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .base_mip_level(0)
                            .base_array_layer(0)
                            .build(),
                    )
                    .src_access_mask(vk::AccessFlags::default())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)],
            );

            // copy staging buffer to image
            device.cmd_copy_buffer_to_image(
                command_buffer,
                self.font_image_staging_buffer,
                self.font_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopyBuilder::new()
                    .image_subresource(
                        vk::ImageSubresourceLayersBuilder::new()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(0)
                            .build(),
                    )
                    .image_extent(
                        vk::Extent3DBuilder::new()
                            .width(dimensions.0 as u32)
                            .height(dimensions.1 as u32)
                            .depth(1)
                            .build(),
                    )],
            );

            // update image layout to shader read only optimal
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrierBuilder::new()
                    .image(self.font_image)
                    .subresource_range(
                        vk::ImageSubresourceRangeBuilder::new()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .base_mip_level(0)
                            .base_array_layer(0)
                            .build(),
                    )
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
            );
        }
         */
    }

    /// Registering user texture.
    ///
    /// Pass the Vulkan ImageView and Sampler.
    /// `image_view`'s image layout must be `SHADER_READ_ONLY_OPTIMAL`.
    ///
    /// UserTexture needs to be unregistered when it is no longer needed.
    ///
    /// # Example
    /// ```sh
    /// cargo run --example user_texture
    /// ```
    /// [The example for user texture is in examples directory](https://github.com/MatchaChoco010/egui_winit_ash_vk_mem_erupt/tree/main/examples/user_texture)
    pub fn register_user_texture(
        &mut self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        rctx: &mut RenderingContext,
    ) -> egui::TextureId {
        // get texture id
        let mut id = None;
        for (i, user_texture) in self.user_textures.iter().enumerate() {
            if user_texture.is_none() {
                id = Some(i as u64);
                break;
            }
        }
        let id = if let Some(i) = id {
            i
        } else {
            self.user_textures.len() as u64
        };

        // allocate and update descriptor set
        let layouts = [self.user_texture_layout];
        let descriptor_set = unsafe {
            rctx.handles.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfoBuilder::new()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .expect("Failed to create descriptor sets.")[0];
        unsafe {
            rctx.handles.device.update_descriptor_sets(
                &[vk::WriteDescriptorSetBuilder::new()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_set(descriptor_set)
                    .image_info(&[vk::DescriptorImageInfoBuilder::new()
                        .image_view(image_view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .sampler(sampler)])
                    .dst_binding(0)],
                &[],
            );
        }

        if id == self.user_textures.len() as u64 {
            self.user_textures.push(Some(descriptor_set));
        } else {
            self.user_textures[id as usize] = Some(descriptor_set);
        }

        egui::TextureId::User(id)
    }

    /// Unregister user texture.
    ///
    /// The internal texture (egui::TextureId::Egui) cannot be unregistered.
    pub fn unregister_user_texture(
        &mut self,
        texture_id: egui::TextureId,
        rctx: &mut RenderingContext,
    ) {
        if let egui::TextureId::User(id) = texture_id {
            if let Some(descriptor_set) = self.user_textures[id as usize] {
                unsafe {
                    rctx.handles
                        .device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .expect("Failed to free descriptor sets.");
                }
                self.user_textures[id as usize] = None;
            }
        } else {
            eprintln!("The internal texture cannot be unregistered; please pass the texture ID of UserTexture.");
        }
    }

    /// destroy vk objects.
    ///
    /// # Safety
    /// This method release vk objects memory that is not managed by Rust.
    pub unsafe fn destroy(&mut self, handles: &RenderingHandles) {
        let device = &handles.device;
        unsafe {
            device.destroy_descriptor_set_layout(self.user_texture_layout, None);
            device.destroy_image_view(self.font_image_view, None);
        }
        let vmalloc = handles.vmalloc.lock();
        vmalloc.destroy_image(self.font_image, &self.font_image_allocation);
        vmalloc.destroy_buffer(
            self.font_image_staging_buffer,
            &self.font_image_staging_buffer_allocation,
        );
        for i in 0..self.index_buffers.len() {
            vmalloc.destroy_buffer(self.index_buffers[i], &self.index_buffer_allocations[i]);
        }
        for i in 0..self.vertex_buffers.len() {
            vmalloc.destroy_buffer(self.vertex_buffers[i], &self.vertex_buffer_allocations[i]);
        }
        drop(vmalloc);
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
        for &descriptor_set_layout in self.descriptor_set_layouts.iter() {
            unsafe {
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
        }
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
