use crate::client::render::vulkan::{allocation_cbs, QueueGuard, RenderingHandles};
use crate::{vk, vk::ObjectHandle};
use bxw_util::*;
use parking_lot::MutexGuard;
use std::ffi::CString;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use vk_mem_3_erupt as vma;

pub fn name_vk_object<F: Fn() -> S, S>(
    handles: &RenderingHandles,
    name_fn: F,
    raw_handle: u64,
    obj_type: vk::ObjectType,
) where
    S: Into<Vec<u8>>,
{
    if handles.ext_debug.is_some() {
        let name_slice = name_fn();
        let name = CString::new(name_slice).unwrap();
        let ni = vk::DebugUtilsObjectNameInfoEXTBuilder::new()
            .object_handle(raw_handle)
            .object_type(obj_type)
            .object_name(name.as_c_str());
        unsafe { handles.device.set_debug_utils_object_name_ext(&ni) }.unwrap();
    }
}

pub trait VulkanDeviceObject {
    fn destroy(&mut self, vmalloc: &mut vma::Allocator, handles: &RenderingHandles);
}

#[derive(Default)]
pub struct OwnedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    /// The identity view for the whole image
    pub allocation: Option<(vma::Allocation, vma::AllocationInfo)>,
}

impl OwnedImage {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from(
        vmalloc: &mut vma::Allocator,
        handles: &RenderingHandles,
        img_info: &vk::ImageCreateInfo,
        mem_info: &vma::AllocationCreateInfo,
        iv_type: vk::ImageViewType,
        iv_aspect: vk::ImageAspectFlags,
    ) -> Self {
        let format = img_info.format;
        let extent = img_info.extent;
        let mip_levels = img_info.mip_levels;
        let array_layers = img_info.array_layers;
        let r = vmalloc
            .create_image(img_info, mem_info)
            .expect("Could not create Vulkan image");
        let ivci = vk::ImageViewCreateInfoBuilder::new()
            .image(r.0)
            .view_type(iv_type)
            .format(img_info.format)
            .components(identity_components())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: iv_aspect,
                base_array_layer: 0,
                base_mip_level: 0,
                layer_count: array_layers,
                level_count: mip_levels,
            });
        let iv = unsafe { handles.device.create_image_view(&ivci, allocation_cbs()) }
            .expect("Couldn't create Vulkan image view");
        Self {
            image: r.0,
            image_view: iv,
            format,
            extent,
            mip_levels,
            array_layers,
            allocation: Some((r.1, r.2)),
        }
    }

    /// Returns the staging buffer which must be destroyed after the command buffer executes.
    #[must_use]
    pub fn upload_bytes(
        &self,
        vmalloc: &mut vma::Allocator,
        handles: &RenderingHandles,
        command_buffer: vk::CommandBuffer,
        copy_desc: vk::BufferImageCopyBuilder,
        bytes: &[u8],
    ) -> OwnedBuffer {
        let staging_buffer = OwnedBuffer::new_single_upload(vmalloc, bytes);
        let device: &erupt::DeviceLoader = &handles.device;
        let subresource_range = vk::ImageSubresourceRangeBuilder::new()
            .aspect_mask(copy_desc.image_subresource.aspect_mask)
            .level_count(1)
            .layer_count(copy_desc.image_subresource.layer_count)
            .base_mip_level(copy_desc.image_subresource.mip_level)
            .base_array_layer(copy_desc.image_subresource.base_array_layer)
            .build();
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrierBuilder::new()
                    .image(self.image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags::default())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)],
            );

            // copy staging buffer to image
            device.cmd_copy_buffer_to_image(
                command_buffer,
                staging_buffer.buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_desc.clone()],
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
                    .image(self.image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
            );
        }
        staging_buffer
    }

    pub fn give_name<F: Fn() -> S, S>(&self, handles: &RenderingHandles, name_fn: F)
    where
        S: Into<Vec<u8>>,
    {
        name_vk_object(
            handles,
            &name_fn,
            self.image.to_raw(),
            vk::ObjectType::IMAGE,
        );
        name_vk_object(
            handles,
            name_fn,
            self.image_view.to_raw(),
            vk::ObjectType::IMAGE_VIEW,
        );
    }
}

impl VulkanDeviceObject for OwnedImage {
    fn destroy(&mut self, vmalloc: &mut vma::Allocator, handles: &RenderingHandles) {
        if self.allocation.is_some() {
            unsafe {
                handles
                    .device
                    .destroy_image_view(self.image_view, allocation_cbs());
            }
            vmalloc.destroy_image(self.image, &self.allocation.take().unwrap().0);
        }
        *self = Self::default();
    }
}

fn byte_slice_from<T: Sized>(data: &[T]) -> &[u8] {
    let bytes_len = data.len() * std::mem::size_of::<T>();
    let start_ptr = data.as_ptr();
    unsafe { std::slice::from_raw_parts(start_ptr as *const u8, bytes_len) }
}

pub struct AllocatorPool(pub vma::AllocatorPool);

unsafe impl Send for AllocatorPool {}

unsafe impl Sync for AllocatorPool {}

impl Clone for AllocatorPool {
    fn clone(&self) -> Self {
        Self(unsafe { std::ptr::read(&self.0 as *const _) })
    }

    fn clone_from(&mut self, source: &Self) {
        unsafe {
            std::ptr::copy_nonoverlapping(&source.0 as *const _, &mut self.0 as *mut _, 1);
        }
    }
}

#[derive(Default)]
pub struct OwnedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<(vma::Allocation, vma::AllocationInfo)>,
    pub size: u64,
}

impl OwnedBuffer {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from(
        vmalloc: &mut vma::Allocator,
        buf_info: &vk::BufferCreateInfo,
        mem_info: &vma::AllocationCreateInfo,
    ) -> Self {
        let r = vmalloc
            .create_buffer(buf_info, mem_info)
            .unwrap_or_else(|e| panic!("Could not create Vulkan buffer: {:#?}", e));
        Self {
            buffer: r.0,
            allocation: Some((r.1, r.2)),
            size: buf_info.size,
        }
    }

    /// Returns a cpu-only, mapped buffer (for one-off uploads) with given contents
    pub fn new_single_upload<T: Sized>(vmalloc: &mut vma::Allocator, data: &[T]) -> Self {
        let data_bytes = byte_slice_from(data);
        let bci = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .size(data_bytes.len() as u64);
        let aci = vma::AllocationCreateInfo {
            usage: vma::MemoryUsage::AutoPreferHost,
            flags: vma::AllocationCreateFlags::MAPPED
                | vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let buf = Self::from(vmalloc, &bci, &aci);
        let (al, ai) = buf.allocation.as_ref().unwrap();
        assert_ne!(ai.get_mapped_data(), std::ptr::null_mut());
        unsafe {
            std::ptr::copy_nonoverlapping(
                data_bytes.as_ptr(),
                ai.get_mapped_data(),
                data_bytes.len(),
            );
        }
        vmalloc.flush_allocation(al, 0, data_bytes.len());
        buf
    }

    pub fn give_name<F: Fn() -> S, S>(&self, handles: &RenderingHandles, name_fn: F)
    where
        S: Into<Vec<u8>>,
    {
        name_vk_object(
            handles,
            name_fn,
            self.buffer.to_raw(),
            vk::ObjectType::BUFFER,
        );
    }
}

impl VulkanDeviceObject for OwnedBuffer {
    fn destroy(&mut self, vmalloc: &mut vma::Allocator, _handles: &RenderingHandles) {
        if self.allocation.is_some() {
            vmalloc.destroy_buffer(self.buffer, &self.allocation.take().unwrap().0);
        }
        *self = Self::default();
    }
}

pub struct FenceGuard<'h> {
    handles: &'h RenderingHandles,
    fence: vk::Fence,
}

impl<'h> FenceGuard<'h> {
    pub fn new<'s, F: FnOnce() -> &'s str>(
        handles: &'h RenderingHandles,
        signaled: bool,
        name_fn: F,
    ) -> Self {
        let fci = vk::FenceCreateInfoBuilder::new().flags(if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        });
        let fence = unsafe { handles.device.create_fence(&fci, allocation_cbs()) }
            .expect("Couldn't create Vulkan fence");
        if handles.ext_debug.is_some() {
            let name_slice = name_fn();
            let name = CString::new(name_slice).unwrap();
            let ni = vk::DebugUtilsObjectNameInfoEXTBuilder::new()
                .object_handle(fence.to_raw())
                .object_type(vk::ObjectType::FENCE)
                .object_name(name.as_c_str());
            unsafe { handles.device.set_debug_utils_object_name_ext(&ni) }.unwrap();
        }
        Self { handles, fence }
    }

    pub fn handle(&self) -> vk::Fence {
        self.fence
    }

    pub fn signaled(&self) -> bool {
        match unsafe { self.handles.device.get_fence_status(self.fence) }.raw {
            vk::Result::SUCCESS => true,
            vk::Result::NOT_READY => false,
            e => panic!("{:?}", e),
        }
    }

    pub fn wait(&self, timeout_ns: Option<u64>) -> erupt::utils::VulkanResult<()> {
        unsafe {
            self.handles.device.wait_for_fences(
                &[self.fence],
                true,
                timeout_ns.unwrap_or(u64::max_value()),
            )
        }
    }

    pub fn reset(&self) {
        unsafe { self.handles.device.reset_fences(&[self.fence]) }.expect("Couldn't reset fence");
    }
}

impl<'h> Drop for FenceGuard<'h> {
    fn drop(&mut self) {
        if self.fence != vk::Fence::null() {
            unsafe {
                self.handles
                    .device
                    .destroy_fence(self.fence, allocation_cbs());
            }
            self.fence = vk::Fence::null();
        }
    }
}

pub struct OnetimeCmdGuard<'h> {
    fence: ManuallyDrop<FenceGuard<'h>>,
    custom_pool: Option<vk::CommandPool>,
    pool_lock: Option<MutexGuard<'h, vk::CommandPool>>,
    cmd: vk::CommandBuffer,
}

impl<'h> OnetimeCmdGuard<'h> {
    pub fn new(handles: &'h RenderingHandles, custom_pool: Option<vk::CommandPool>) -> Self {
        let (pool_lock, pool) = if let Some(pool) = custom_pool {
            (None, pool)
        } else {
            let lock = handles.oneoff_cmd_pool.lock();
            let pool = *lock;
            (Some(lock), pool)
        };
        let bai = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { handles.device.allocate_command_buffers(&bai) }
            .expect("Couldn't allocate one-time cmd buffer")[0];
        let bgi = vk::CommandBufferBeginInfoBuilder::new()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { handles.device.begin_command_buffer(cmd, &bgi) }
            .expect("Couldn't begin recording one-time cmd buffer");
        Self {
            fence: ManuallyDrop::new(FenceGuard::new(handles, false, || "one-time cmd")),
            custom_pool,
            pool_lock,
            cmd,
        }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.cmd
    }

    pub fn execute(mut self, queue: &QueueGuard) {
        let handles = self.fence.handles;
        unsafe { handles.device.end_command_buffer(self.cmd) }
            .expect("Couldn't end recording one-time cmd buffer");

        let bufs = [self.cmd];

        let pool_lock = self.pool_lock.take();
        let sis = [vk::SubmitInfoBuilder::new().command_buffers(&bufs)];
        unsafe { handles.device.queue_submit(**queue, &sis, self.fence.fence) }
            .expect("Couldn't submit one-time cmd buffer");
        self.fence
            .wait(None)
            .expect("Failed waiting for one-time cmd fence");
        unsafe {
            ManuallyDrop::drop(&mut self.fence);
        }

        unsafe {
            handles.device.free_command_buffers(
                pool_lock.map_or_else(|| self.custom_pool.unwrap(), |p| *p),
                &bufs,
            );
        }
        std::mem::forget(self);
    }
}

impl<'h> Drop for OnetimeCmdGuard<'h> {
    fn drop(&mut self) {
        panic!("One-time command buffer not invoked");
    }
}

#[derive(Copy, Clone, Default)]
pub struct DynamicState {
    pub viewport: vk::Viewport,
}

impl DynamicState {
    pub fn get_viewport(&self) -> vk::Viewport {
        self.viewport
    }

    pub fn get_scissor(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: Default::default(),
            extent: vk::Extent2D {
                width: self.viewport.width as u32,
                height: self.viewport.height as u32,
            },
        }
    }

    pub fn cmd_update_pipeline(&self, device: &erupt::DeviceLoader, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_set_viewport(cmd, 0, &[self.get_viewport().into_builder()]);
            device.cmd_set_scissor(cmd, 0, &[self.get_scissor().into_builder()]);
        }
    }
}

pub fn make_pipe_depthstencil() -> vk::PipelineDepthStencilStateCreateInfo {
    vk::PipelineDepthStencilStateCreateInfoBuilder::new()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .build_dangling()
}

pub fn identity_components() -> vk::ComponentMapping {
    vk::ComponentMapping {
        r: vk::ComponentSwizzle::IDENTITY,
        g: vk::ComponentSwizzle::IDENTITY,
        b: vk::ComponentSwizzle::IDENTITY,
        a: vk::ComponentSwizzle::IDENTITY,
    }
}

pub fn cmd_push_struct_constants<S>(
    device: &erupt::DeviceLoader,
    cmd: vk::CommandBuffer,
    pipe_layout: vk::PipelineLayout,
    stages: vk::ShaderStageFlags,
    offset: u32,
    constants: &S,
) {
    let pc_sz = std::mem::size_of::<S>();
    let pc_bytes = unsafe { std::slice::from_raw_parts(constants as *const _ as *const u8, pc_sz) };
    unsafe {
        device.cmd_push_constants(
            cmd,
            pipe_layout,
            stages,
            offset,
            pc_bytes.len() as u32,
            pc_bytes.as_ptr() as *const std::ffi::c_void,
        );
    }
}

pub struct DroppingCommandPool {
    pub pool: vk::CommandPool,
    device: Arc<erupt::DeviceLoader>,
}

impl DroppingCommandPool {
    pub fn new(handles: &RenderingHandles, pci: &vk::CommandPoolCreateInfo) -> Self {
        let pool = unsafe { handles.device.create_command_pool(pci, allocation_cbs()) }
            .expect("Couldn't create command pool");
        Self {
            pool,
            device: handles.device.clone(),
        }
    }
}

impl Drop for DroppingCommandPool {
    fn drop(&mut self) {
        if self.pool != vk::CommandPool::null() {
            unsafe {
                self.device
                    .destroy_command_pool(self.pool, allocation_cbs());
            }
            self.pool = vk::CommandPool::null();
        }
    }
}

#[derive(Default)]
pub struct OwnedDescriptorSet(pub vk::DescriptorPool, pub vk::DescriptorSet);

impl OwnedDescriptorSet {
    pub fn from(
        handles: &RenderingHandles,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
    ) -> Self {
        let ds = unsafe {
            handles
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfoBuilder::new()
                        .descriptor_pool(pool)
                        .set_layouts(&[layout]),
                )
                .expect("Couldn't allocate descriptor sets")[0]
        };
        Self(pool, ds)
    }
}

impl VulkanDeviceObject for OwnedDescriptorSet {
    fn destroy(&mut self, _vmalloc: &mut vma::Allocator, handles: &RenderingHandles) {
        if !self.0.is_null() && !self.1.is_null() {
            unsafe { handles.device.free_descriptor_sets(self.0, &[self.1]) }
                .expect("Couldn't free descriptor set");
        }
        *self = Self::default();
    }
}
