use sdl2::video::Window;
use std::ffi::CString;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Queue};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, RawInstanceExtensions};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain::{
    AcquireError, PresentMode, Surface, SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::sync::{FlushError, GpuFuture};

pub mod vox {
    pub struct ChunkBuffers {
        pub vertices: Vec<VoxelVertex>,
        pub indices: Vec<u32>
    }

    #[derive(Copy, Clone, Default)]
    pub struct VoxelVertex {
        pub position: [f32; 4],
        pub color: [f32; 4],
    }
    impl_vertex!(VoxelVertex, position, color);

    pub mod vs {
        vulkano_shaders::shader! {
        ty: "vertex",
            src: "
#version 450

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 v_color;

void main() {
    gl_Position = vec4(position.xyz, 1.0);
    v_color = color;
}"
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
        ty: "fragment",
            src: "
#version 450

layout(location = 0) in vec4 v_color;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = v_color;
}"
        }
    }
}

pub enum Queues {
    /// Combined Graphics+Transfer+Compute queue
    Combined(Arc<Queue>),
}

pub struct RenderingContext {
    pub window: Window,
    pub instance_extensions: InstanceExtensions,
    pub instance: Arc<Instance>,
    pub surface: Arc<Surface<()>>,
    pub device: Arc<Device>,
    pub queues: Queues,
    pub swapchain: Arc<Swapchain<()>>,
    pub swapimages: Vec<Arc<SwapchainImage<()>>>,
    pub mainpass: Arc<RenderPassAbstract + Send + Sync>,
    pub framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    pub dynamic_state: DynamicState,
    pub previous_frame_end: Box<GpuFuture>,
    pub outdated_swapchain: bool,
    pub voxel_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub vbuffer: Arc<CpuAccessibleBuffer<[vox::VoxelVertex]>>,
}

fn generate_updated_framebuffers(
    swapimages: &[Arc<SwapchainImage<()>>],
    mainpass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = swapimages[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    swapimages
        .iter()
        .map(|img| {
            Arc::new(
                Framebuffer::start(mainpass.clone())
                    .add(img.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

impl RenderingContext {
    pub fn new(sdl_video: &sdl2::VideoSubsystem) -> RenderingContext {
        sdl_video.vulkan_load_library_default().unwrap();
        let window = sdl_video
            .window("BallX World", 1280, 720)
            .position_centered()
            .vulkan()
            .allow_highdpi()
            .resizable()
            .build()
            .expect("Failed to create the game window");
        let instance_extensions;
        let instance = {
            let sdl_exts = window
                .vulkan_instance_extensions()
                .expect("Couldn't get a list of the required VK instance extensions");
            let raw_exts = RawInstanceExtensions::new(
                sdl_exts
                    .into_iter()
                    .map(|x| CString::new(x).expect("Invalid required VK instance extension")),
            );
            instance_extensions = InstanceExtensions::from(&raw_exts);
            Instance::new(None, &instance_extensions, None)
        }
        .expect("Failed to create Vulkan instance");
        let surface = Arc::new(unsafe {
            use vulkano::VulkanObject;
            let sdlvki = instance.internal_object() as sdl2::video::VkInstance;
            let rsurf = window
                .vulkan_create_surface(sdlvki)
                .expect("Couldn't create VK surface") as u64;
            Surface::from_raw_surface(instance.clone(), rsurf, ())
        });
        let physical = PhysicalDevice::enumerate(&instance)
            .next()
            .expect("no device available");

        println!("Choosing device {}", physical.name());
        for family in physical.queue_families() {
            println!(
                "Found a queue family with {:?} queue(s)",
                family.queues_count()
            );
        }

        let queue_family = physical
            .queue_families()
            .find(|&q| {
                q.supports_graphics()
                    && q.supports_compute()
                    && surface.is_supported(q).unwrap_or(false)
            })
            .expect("couldn't find a graphical queue family");
        let (device, mut queues) = {
            let device_ext = vulkano::device::DeviceExtensions {
                khr_swapchain: true,
                ..vulkano::device::DeviceExtensions::none()
            };
            Device::new(
                physical,
                physical.supported_features(),
                &device_ext,
                [(queue_family, 0.5)].iter().cloned(),
            )
            .expect("failed to create device")
        };
        let queue = queues.next().expect("Couldn't create rendering queue");

        let (swapchain, swapimages) = {
            let caps = surface
                .capabilities(physical)
                .expect("failed to get surface capabilities");
            let dimensions = caps.current_extent.unwrap_or([1280, 720]);
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;
            Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                dimensions,
                1,
                caps.supported_usage_flags,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Fifo,
                true,
                None,
            )
            .expect("failed to create swapchain")
        };

        let mainpass = Arc::new(
            single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        let mut dynamic_state = DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
        };

        let framebuffers =
            generate_updated_framebuffers(&swapimages, mainpass.clone(), &mut dynamic_state);

        let previous_frame_end = Box::new(vulkano::sync::now(device.clone()));

        let vbuffer = {
            let v1 = vox::VoxelVertex {
                position: [-0.5, -0.5, 0.0, 0.0],
                color: [1.0, 0.0, 0.0, 1.0],
            };
            let v2 = vox::VoxelVertex {
                position: [0.0, 0.5, 0.0, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
            };
            let v3 = vox::VoxelVertex {
                position: [0.5, -0.25, 0.0, 0.0],
                color: [0.0, 0.0, 1.0, 1.0],
            };
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::vertex_buffer(),
                vec![v1, v2, v3].into_iter(),
            )
            .unwrap()
        };

        let voxel_pipeline = {
            let vs = vox::vs::Shader::load(device.clone()).expect("Failed to create VS module");
            let fs = vox::fs::Shader::load(device.clone()).expect("Failed to create FS module");
            Arc::new(
                GraphicsPipeline::start()
                    .vertex_input_single_buffer::<vox::VoxelVertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .viewports_dynamic_scissors_irrelevant(1)
                    .fragment_shader(fs.main_entry_point(), ())
                    .render_pass(Subpass::from(mainpass.clone(), 0).unwrap())
                    .build(device.clone())
                    .expect("Could not create voxel graphics pipeline"),
            )
        };

        RenderingContext {
            window,
            instance_extensions,
            instance,
            device,
            surface,
            queues: Queues::Combined(queue),
            swapchain,
            swapimages,
            mainpass,
            framebuffers,
            dynamic_state,
            previous_frame_end,
            outdated_swapchain: false,
            voxel_pipeline,
            vbuffer,
        }
    }

    pub fn draw_next_frame(&mut self, _delta_time: f64) {
        self.previous_frame_end.cleanup_finished();

        if self.outdated_swapchain {
            let dims = {
                let d = self.window.vulkan_drawable_size();
                [d.0, d.1]
            };

            let (new_swapchain, new_images) = match self.swapchain.recreate_with_dimension(dims) {
                Ok(r) => r,
                // occurs on manual resizes, just ignore
                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                Err(err) => panic!("{:?}", err),
            };

            self.swapchain = new_swapchain;
            self.swapimages = new_images;
            self.framebuffers = generate_updated_framebuffers(
                &self.swapimages,
                self.mainpass.clone(),
                &mut self.dynamic_state,
            );

            self.outdated_swapchain = false;
        }

        let Queues::Combined(ref queue) = self.queues;
        let (image_num, acquire_future) =
            match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.outdated_swapchain = true;
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        let cmdbuf =
            AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), queue.family())
                .unwrap()
                .begin_render_pass(
                    self.framebuffers[image_num].clone(),
                    false,
                    vec![[0.1, 0.1, 0.1, 1.0].into()],
                )
                .unwrap()
                //
                .draw(
                    self.voxel_pipeline.clone(),
                    &self.dynamic_state,
                    vec![self.vbuffer.clone()],
                    (),
                    (),
                )
                .unwrap()
                //
                .end_render_pass()
                .unwrap()
                //
                .build()
                .unwrap();
        let myfuture = std::mem::replace(
            &mut self.previous_frame_end,
            Box::new(vulkano::sync::now(self.device.clone())),
        );

        let future = myfuture
            .join(acquire_future)
            .then_execute(queue.clone(), cmdbuf)
            .unwrap()
            .then_swapchain_present(queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();
        match future {
            Ok(future) => {
                self.previous_frame_end = Box::new(future);
            }
            Err(FlushError::OutOfDate) => {
                self.outdated_swapchain = true;
            }
            Err(e) => {
                println!("{:?}", e);
            }
        }
    }
}
