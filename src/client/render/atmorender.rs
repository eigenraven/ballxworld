use crate::client::render::vkhelpers::{cmd_push_struct_constants, make_pipe_depthstencil};
use crate::client::render::vulkan::{allocation_cbs, RenderingHandles};
use crate::client::render::{InPassFrameContext, RenderingContext};
use crate::config::Config;
use crate::vk;
use crate::vk::ObjectHandle;
use bxw_util::math::*;
use std::ffi::CString;

pub mod shaders {
    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    pub struct SkyUBO {
        pub viewproj: [[f32; 4]; 4],
    }
}

pub struct AtmosphereRenderer {
    pub sky_pipeline_layout: vk::PipelineLayout,
    pub sky_pipeline: vk::Pipeline,
}

impl AtmosphereRenderer {
    pub fn new(_cfg: &Config, rctx: &mut RenderingContext) -> Self {
        let pipeline_layout = {
            let pc = [vk::PushConstantRangeBuilder::new()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(4 * 4 * 4)];
            let dsls = [];
            let lci = vk::PipelineLayoutCreateInfoBuilder::new()
                .set_layouts(&dsls)
                .push_constant_ranges(&pc);
            unsafe {
                rctx.handles
                    .device
                    .create_pipeline_layout(&lci, allocation_cbs())
            }
            .expect("Could not create voxel pipeline layout")
        };

        // create pipeline
        let sky_pipeline = {
            let vs = rctx
                .handles
                .load_shader_module("res/shaders/atmosphere.vert.spv")
                .expect("Couldn't load vertex atmosphere shader");
            let fs = rctx
                .handles
                .load_shader_module("res/shaders/atmosphere.frag.spv")
                .expect("Couldn't load fragment atmosphere shader");
            //
            let cmain = CString::new("main").unwrap();
            let vss = vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::VERTEX)
                .module(vs)
                .name(&cmain);
            let fss = vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::FRAGMENT)
                .module(fs)
                .name(&cmain);
            let shaders = [vss, fss];
            let vtxinp = vk::PipelineVertexInputStateCreateInfoBuilder::new();
            let inpasm = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);
            let viewport = [rctx.swapchain.dynamic_state.get_viewport().into_builder()];
            let scissor = [rctx.swapchain.dynamic_state.get_scissor().into_builder()];
            let vwp_info = vk::PipelineViewportStateCreateInfoBuilder::new()
                .scissors(&scissor)
                .viewports(&viewport);
            let raster = vk::PipelineRasterizationStateCreateInfoBuilder::new()
                .depth_clamp_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);
            let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
                .rasterization_samples(rctx.handles.sample_count)
                .sample_shading_enable(true)
                .min_sample_shading(0.2);
            let depthstencil = make_pipe_depthstencil();
            let blendings = [vk::PipelineColorBlendAttachmentStateBuilder::new()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)];
            let blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
                .logic_op_enable(false)
                .attachments(&blendings)
                .blend_constants([0.0, 0.0, 0.0, 0.0]);
            let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dyn_state =
                vk::PipelineDynamicStateCreateInfoBuilder::new().dynamic_states(&dyn_states);

            let pcis = [vk::GraphicsPipelineCreateInfoBuilder::new()
                .stages(&shaders)
                .vertex_input_state(&vtxinp)
                .input_assembly_state(&inpasm)
                .viewport_state(&vwp_info)
                .rasterization_state(&raster)
                .multisample_state(&multisampling)
                .depth_stencil_state(&depthstencil)
                .color_blend_state(&blending)
                .dynamic_state(&dyn_state)
                .layout(pipeline_layout)
                .render_pass(rctx.handles.mainpass)
                .subpass(0)
                .base_pipeline_handle(vk::Pipeline::null())
                .base_pipeline_index(-1)];

            let pipeline = unsafe {
                rctx.handles.device.create_graphics_pipelines(
                    rctx.pipeline_cache,
                    &pcis,
                    allocation_cbs(),
                )
            }
            .expect("Could not create atmosphere pipeline")[0];

            unsafe {
                rctx.handles
                    .device
                    .destroy_shader_module(vs, allocation_cbs());
                rctx.handles
                    .device
                    .destroy_shader_module(fs, allocation_cbs());
            }
            pipeline
        };

        Self {
            sky_pipeline_layout: pipeline_layout,
            sky_pipeline,
        }
    }

    pub fn destroy(&self, handles: &RenderingHandles) {
        unsafe {
            handles
                .device
                .destroy_pipeline(self.sky_pipeline, allocation_cbs());
            handles
                .device
                .destroy_pipeline_layout(self.sky_pipeline_layout, allocation_cbs());
        }
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn inpass_draw(
        &mut self,
        fctx: &mut InPassFrameContext,
        mview: Matrix4<f32>,
        mproj: Matrix4<f32>,
    ) {
        let ubo = {
            let mut ubo = shaders::SkyUBO::default();
            let proj = mproj;
            let mut view = mview;
            view.set_column(3, &vec4(0.0, 0.0, 0.0, 1.0));
            ubo.viewproj = (proj * view).into();
            ubo
        };

        let device = &fctx.rctx.handles.device;

        unsafe {
            device.cmd_bind_pipeline(fctx.cmd, vk::PipelineBindPoint::GRAPHICS, self.sky_pipeline);
        }
        fctx.rctx
            .swapchain
            .dynamic_state
            .cmd_update_pipeline(device, fctx.cmd);

        cmd_push_struct_constants(
            device,
            fctx.cmd,
            self.sky_pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &ubo,
        );
        unsafe {
            device.cmd_draw(fctx.cmd, 36, 1, 0, 0);
        }
    }
}
