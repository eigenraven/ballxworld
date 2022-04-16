use crate::client::render::vkhelpers::{
    identity_components, DynamicState, OwnedImage, VulkanDeviceObject,
};
use crate::client::world::ClientWorld;
use crate::config::Config;
use crate::vk;
use bxw_util::debug_data::DEBUG_DATA;
use bxw_util::*;
use erupt::vk::EXT_DEBUG_UTILS_EXTENSION_NAME;
use erupt::{DeviceLoader, DeviceLoaderBuilder, EntryLoader, InstanceLoader, InstanceLoaderBuilder};
use num_traits::clamp;
use parking_lot::{Mutex, MutexGuard};
use sdl2::video::Window;
use std::cmp::max;
use std::ffi::{c_void, CStr, CString};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::os::raw::c_char;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use vk_mem_erupt as vma;
use bxw_util::fnv::FnvHashMap;
use bxw_util::parking_lot::lock_api::MappedMutexGuard;
use bxw_util::parking_lot::RawMutex;

static VULK_ALLOCATIONS: Mutex<Option<fnv::FnvHashMap<usize, std::alloc::Layout>>> = Mutex::new(None);

fn vulk_allocations() -> MappedMutexGuard<'static, RawMutex, FnvHashMap<usize, std::alloc::Layout>> {
    MutexGuard::map(VULK_ALLOCATIONS.lock(), |v| v.get_or_insert_with(Default::default))
}

unsafe extern "system" fn vk_reallocate(
    _p_user_data: *mut std::ffi::c_void,
    p_original: *mut std::ffi::c_void,
    size: usize,
    alignment: usize,
    _allocation_scope: vk::SystemAllocationScope,
) -> *mut std::ffi::c_void {
    let original_layout = vulk_allocations().remove(&(p_original as usize)).expect("Vulkan trying to free memory without matching allocation");
    let new_layout = std::alloc::Layout::from_size_align(size, alignment).unwrap();
    let new_ptr = if alignment == original_layout.align() { unsafe {
        std::alloc::realloc(p_original as *mut u8, original_layout, size) as *mut c_void
    }} else {
        unsafe {
            let new_ptr = std::alloc::alloc_zeroed(new_layout);
            if new_ptr == std::ptr::null_mut() {
                panic!("Couldn't allocate memory for Vulkan");
            }
            std::ptr::copy_nonoverlapping(p_original as *mut u8, new_ptr, original_layout.size().min(size));
            std::alloc::dealloc(p_original as *mut u8, original_layout);
            new_ptr as *mut c_void
        }
    };
    vulk_allocations().insert(new_ptr as usize, new_layout);
    new_ptr
}

unsafe extern "system" fn vk_allocate(
    _p_user_data: *mut std::ffi::c_void,
    size: usize,
    alignment: usize,
    _allocation_scope: vk::SystemAllocationScope,
) -> *mut std::ffi::c_void {
    let layout = std::alloc::Layout::from_size_align(size, alignment).unwrap();
    let ptr = unsafe {
        std::alloc::alloc_zeroed(layout) as *mut c_void
    };
    vulk_allocations().insert(ptr as usize, layout);
    ptr
}

unsafe extern "system" fn vk_free(
    _p_user_data: *mut std::ffi::c_void,
    p_memory: *mut std::ffi::c_void,
) {
    let layout = vulk_allocations().remove(&(p_memory as usize)).expect("Vulkan trying to free memory without matching allocation");
    unsafe {
        std::alloc::dealloc(p_memory as *mut u8, layout);
    }
}

struct AllocationCallbacksHolder(vk::AllocationCallbacks);
// Safety: these are statically initialized, non-changing function pointers
unsafe impl Send for AllocationCallbacksHolder {}
// Safety: these are statically initialized, non-changing function pointers
unsafe impl Sync for AllocationCallbacksHolder {}

static ALLOCATION_CBS: AllocationCallbacksHolder = AllocationCallbacksHolder(vk::AllocationCallbacks {
    p_user_data: std::ptr::null_mut(),
    pfn_allocation: Some(vk_allocate),
    pfn_reallocation: Some(vk_reallocate),
    pfn_free: Some(vk_free),
    pfn_internal_allocation: None,
    pfn_internal_free: None
});

pub fn allocation_cbs() -> Option<&'static vk::AllocationCallbacks> {
    Some(&ALLOCATION_CBS.0)
}

pub type QueueGuard<'a> = MutexGuard<'a, vk::Queue>;
/// (queue, family)
pub type QueuePair = (Mutex<vk::Queue>, u32);

pub enum Queues {
    /// Combined Graphics+Transfer+Compute queue
    Combined(QueuePair),
    /// Two queues from the same G+T+C family
    Dual {
        render: QueuePair,
        gtransfer: QueuePair,
    },
}

impl Queues {
    pub fn get_primary_family(&self) -> u32 {
        match self {
            Queues::Combined(q) => q.1,
            Queues::Dual { render, .. } => render.1,
        }
    }

    pub fn get_gtransfer_family(&self) -> u32 {
        match self {
            Queues::Combined(q) => q.1,
            Queues::Dual { gtransfer, .. } => gtransfer.1,
        }
    }

    pub fn lock_primary_queue(&self) -> QueueGuard {
        match self {
            Queues::Combined(q) => q.0.lock(),
            Queues::Dual { render, .. } => render.0.lock(),
        }
    }

    /// Warning: might lock main queue if gtransfer not present!
    pub fn lock_gtransfer_queue(&self) -> QueueGuard {
        match self {
            Queues::Combined(q) => q.0.lock(),
            Queues::Dual { gtransfer, .. } => gtransfer.0.lock(),
        }
    }
}

pub const INFLIGHT_FRAMES: u32 = 2;

#[derive(Clone)]
pub struct DebugExts {
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
}

/// Vector of queues for object destruction, one per inflight frame
pub type VDODestroyQueue = Vec<Box<dyn VulkanDeviceObject + Send>>;

#[derive(Clone)]
pub struct RenderingHandles {
    pub entry: Arc<EntryLoader>,
    pub instance: Arc<InstanceLoader>,
    pub ext_debug: Option<DebugExts>,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub physical: vk::PhysicalDevice,
    pub physical_limits: vk::PhysicalDeviceLimits,
    pub sample_count: vk::SampleCountFlagBits,
    pub device: Arc<DeviceLoader>,
    pub queues: Arc<Queues>,
    pub vmalloc: Arc<Mutex<vma::Allocator>>,
    pub mainpass: vk::RenderPass,
    pub oneoff_cmd_pool: Arc<Mutex<vk::CommandPool>>,
    pub inflight_index: u32,
    pub destroy_queue: Arc<Mutex<VDODestroyQueue>>,
    pub frame_destroy_queues: Arc<Mutex<Vec<VDODestroyQueue>>>,
}

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub swapimage_size: vk::Extent2D,
    pub swapimages: vk::SmallVec<vk::Image>,
    pub swapimageviews: Vec<vk::ImageView>,
    pub depth_image: OwnedImage,
    pub color_image: OwnedImage,
    pub inflight_render_finished_semaphores: Vec<vk::Semaphore>,
    pub inflight_image_available_semaphores: Vec<vk::Semaphore>,
    pub inflight_fences: Vec<vk::Fence>,
    pub outdated: bool,
    pub dynamic_state: DynamicState,
    pub framebuffers: Vec<vk::Framebuffer>,
}

pub struct RenderingContext {
    pub window: Window,
    // basic handles
    pub handles: RenderingHandles,
    // swapchain
    pub swapchain: ManuallyDrop<Swapchain>,
    frame_index: u32,
    // default render set
    pub cmd_pool: vk::CommandPool,
    pub inflight_cmds: vk::SmallVec<vk::CommandBuffer>,
    pub pipeline_cache: vk::PipelineCache,
}

pub trait FrameStage {}

pub struct PrePassStage {}

impl FrameStage for PrePassStage {}

pub struct InPassStage {}

impl FrameStage for InPassStage {}

pub struct PostPassStage {}

impl FrameStage for PostPassStage {}

#[must_use]
pub struct FrameContext<'r, 'cw, Stage: FrameStage> {
    pub rctx: &'r mut RenderingContext,
    pub cmd: vk::CommandBuffer,
    pub client_world: &'cw ClientWorld,
    pub delta_time: f64,
    pub dims: [u32; 2],
    pub image_index: usize,
    pub inflight_index: usize,
    _phantom: PhantomData<Stage>,
}

impl<'r, 'cw, Stage: FrameStage> FrameContext<'r, 'cw, Stage> {
    pub fn begin_region<F: FnOnce() -> S, S>(&self, color: [f32; 4], name_fn: F)
    where
        S: Into<Vec<u8>>,
    {
        if self.rctx.handles.ext_debug.is_some() {
            let name_slice = name_fn();
            let name = CString::new(name_slice).unwrap();
            let label = vk::DebugUtilsLabelEXTBuilder::new()
                .color(color)
                .label_name(name.as_c_str());
            unsafe {
                self.rctx
                    .handles
                    .device
                    .cmd_begin_debug_utils_label_ext(self.cmd, &label);
            }
        }
    }

    pub fn end_region(&self) {
        if self.rctx.handles.ext_debug.is_some() {
            unsafe {
                self.rctx
                    .handles
                    .device
                    .cmd_end_debug_utils_label_ext(self.cmd);
            }
        }
    }

    pub fn insert_label<F: FnOnce() -> S, S>(&self, color: [f32; 4], name_fn: F)
    where
        S: Into<Vec<u8>>,
    {
        if self.rctx.handles.ext_debug.is_some() {
            let name_slice = name_fn();
            let name = CString::new(name_slice).unwrap();
            let label = vk::DebugUtilsLabelEXTBuilder::new()
                .color(color)
                .label_name(name.as_c_str());
            unsafe {
                self.rctx
                    .handles
                    .device
                    .cmd_insert_debug_utils_label_ext(self.cmd, &label);
            }
        }
    }
}

pub type PrePassFrameContext<'r, 'cw> = FrameContext<'r, 'cw, PrePassStage>;
pub type InPassFrameContext<'r, 'cw> = FrameContext<'r, 'cw, InPassStage>;
pub type PostPassFrameContext<'r, 'cw> = FrameContext<'r, 'cw, PostPassStage>;

#[allow(unused_unsafe)]
unsafe extern "system" fn debug_msg_callback(
    msg_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _udata: *mut c_void,
) -> vk::Bool32 {
    if cb_data.is_null() {
        return vk::FALSE;
    }
    if msg_severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::VERBOSE_EXT {
        return vk::FALSE;
    }
    let cb_data: &vk::DebugUtilsMessengerCallbackDataEXT = unsafe { &*cb_data };
    use log::Level;
    let log_severity = match msg_severity {
        vk::DebugUtilsMessageSeverityFlagBitsEXT::VERBOSE_EXT => Level::Debug,
        vk::DebugUtilsMessageSeverityFlagBitsEXT::INFO_EXT => Level::Info,
        vk::DebugUtilsMessageSeverityFlagBitsEXT::WARNING_EXT => Level::Warn,
        vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT => Level::Error,
        _ => Level::Warn,
    };
    let str_msg = format!(
        "{} [#{} {}]",
        unsafe { CStr::from_ptr(cb_data.p_message) }.to_string_lossy(),
        cb_data.message_id_number,
        unsafe { CStr::from_ptr(cb_data.p_message_id_name) }.to_string_lossy(),
    );
    log::log!(target: "vulkan", log_severity, "{}", str_msg);
    vk::FALSE
}

impl RenderingHandles {
    fn new(sdl_video: &sdl2::VideoSubsystem, cfg: &Config) -> (Window, Self) {
        sdl_video.vulkan_load_library_default().unwrap();
        let mut window = sdl_video.window("BallX World", cfg.window_width, cfg.window_height);
        window
            .position_centered()
            .vulkan()
            .allow_highdpi()
            .resizable();
        if cfg.window_fullscreen {
            window.fullscreen();
        }
        let window = window.build().expect("Failed to create the game window");
        let entry = Arc::new(
            erupt::EntryLoader::new().expect("Can't load Vulkan system library entrypoints"),
        );
        if entry.instance_version() < vk::make_api_version(0, 1, 2, 0) {
            panic!("Vulkan 1.2 not found");
        }
        let (instance, ext_debug) = Self::create_instance(&entry, &window, cfg);
        let surface = {
            let sdlvki = instance.handle.object_handle() as sdl2::video::VkInstance;
            let rsurf = window
                .vulkan_create_surface(sdlvki)
                .expect("Couldn't create VK surface") as u64;
            vk::SurfaceKHR(rsurf)
        };
        let physical = unsafe { instance.enumerate_physical_devices(None) }
            .expect("Could not enumerate Vulkan physical devices")
            .into_iter()
            .next()
            .expect("no device available");
        let pprop = unsafe { instance.get_physical_device_properties(physical) };
        let physical_limits = pprop.limits;
        let pname = unsafe { CStr::from_ptr(pprop.device_name.as_ptr()) }.to_string_lossy();

        log::info!("Choosing device {}", pname);
        let qfamilies =
            unsafe { instance.get_physical_device_queue_family_properties(physical, None) };
        for family in qfamilies.iter() {
            log::debug!(
                "Found a queue family with {:?} queue(s)",
                family.queue_count
            );
        }

        let queue_family = qfamilies
            .iter()
            .enumerate()
            .find(|(i, &q)| {
                q.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                    && unsafe {
                        instance
                            .get_physical_device_surface_support_khr(physical, *i as u32, surface)
                            .result()
                            .unwrap_or(false)
                    }
            })
            .map(|(i, q)| (i as u32, q))
            .expect("couldn't find a graphical queue family");
        let queue_cnt;
        let device = {
            let raw_exts: Vec<&'static CStr> =
                vec![unsafe { CStr::from_ptr(vk::KHR_SWAPCHAIN_EXTENSION_NAME) }];
            let exts: Vec<*const c_char> = raw_exts.iter().map(|s| s.as_ptr()).collect();
            let avail_exts =
                unsafe { instance.enumerate_device_extension_properties(physical, None, None) }
                    .expect("Can't enumerate VkDevice extensions");
            for ext in raw_exts.iter() {
                let available = avail_exts.iter().any(|e| {
                    let en = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                    en == *ext
                });
                if !available {
                    panic!(
                        "Required Vulkan Device extension {} not present on selected device!",
                        ext.to_str().unwrap()
                    );
                }
            }
            let features = vk::PhysicalDeviceFeaturesBuilder::new()
                .sampler_anisotropy(true)
                .sample_rate_shading(true);

            queue_cnt = queue_family.1.queue_count.min(2);
            let priorities: Vec<std::os::raw::c_float> = if queue_cnt == 1 {
                vec![1.0]
            } else {
                vec![0.75, 0.25]
            };
            let queue_families = vec![vk::DeviceQueueCreateInfoBuilder::new()
                .queue_family_index(queue_family.0)
                .queue_priorities(&priorities)];

            let dci = vk::DeviceCreateInfoBuilder::new()
                .enabled_extension_names(&exts)
                .enabled_features(&features)
                .queue_create_infos(&queue_families);

            Arc::new(
                unsafe { DeviceLoaderBuilder::new().allocation_callbacks(allocation_cbs().unwrap()).build(&instance, physical, &dci) }
                    .expect("Couldn't create Vulkan device"),
            )
        };
        let queues = if queue_cnt > 1 {
            if cfg.debug_logging {
                log::info!("Creating 2 Vulkan queues for asynchronous operations");
            }
            let rqueue = unsafe { device.get_device_queue(queue_family.0, 0) };
            let tqueue = unsafe { device.get_device_queue(queue_family.0, 1) };
            Queues::Dual {
                render: (Mutex::new(rqueue), queue_family.0),
                gtransfer: (Mutex::new(tqueue), queue_family.0),
            }
        } else {
            if cfg.debug_logging {
                log::info!("Creating 1 Vulkan queue - more are not supported");
            }
            let rqueue = unsafe { device.get_device_queue(queue_family.0, 0) };
            Queues::Combined((Mutex::new(rqueue), queue_family.0))
        };

        let vmalloc = {
            let ai = vma::AllocatorCreateInfo {
                flags: vma::AllocatorCreateFlags::EXTERNALLY_SYNCHRONIZED,
                instance: instance.clone(),
                physical_device: physical,
                device: device.clone(),
                frame_in_use_count: INFLIGHT_FRAMES,
                heap_size_limits: None,
                preferred_large_heap_block_size: 0,
            };
            vma::Allocator::new(&ai).expect("Could not create Vulkan memory allocator")
        };

        let formats =
            unsafe { instance.get_physical_device_surface_formats_khr(physical, surface, None) }
                .expect("Failed to get surface formats");
        let surface_format = *formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
            })
            .or_else(|| {
                formats.iter().find(|f| {
                    f.format == vk::Format::R8G8B8A8_UNORM
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
                })
            })
            .unwrap_or(&formats[0]);

        let sample_count = {
            let target = cfg.render_samples;
            let mut samples = vk::SampleCountFlagBits::_1;
            let lim = physical_limits.framebuffer_color_sample_counts
                & physical_limits.sampled_image_depth_sample_counts;
            if target >= 64 && lim.contains(vk::SampleCountFlags::_64) {
                samples = vk::SampleCountFlagBits::_64;
            }
            if target >= 32 && lim.contains(vk::SampleCountFlags::_32) {
                samples = vk::SampleCountFlagBits::_32;
            }
            if target >= 16 && lim.contains(vk::SampleCountFlags::_16) {
                samples = vk::SampleCountFlagBits::_16;
            }
            if target >= 8 && lim.contains(vk::SampleCountFlags::_8) {
                samples = vk::SampleCountFlagBits::_8;
            }
            if target >= 4 && lim.contains(vk::SampleCountFlags::_4) {
                samples = vk::SampleCountFlagBits::_4;
            }
            if target >= 2 && lim.contains(vk::SampleCountFlags::_2) {
                samples = vk::SampleCountFlagBits::_2;
            }
            samples
        };

        let mainpass = {
            let color_at = vk::AttachmentDescriptionBuilder::new()
                .format(surface_format.format)
                .samples(sample_count)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let depth_at = vk::AttachmentDescriptionBuilder::new()
                .format(vk::Format::D32_SFLOAT_S8_UINT)
                .samples(sample_count)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            let resolve_at = vk::AttachmentDescriptionBuilder::new()
                .format(surface_format.format)
                .samples(vk::SampleCountFlagBits::_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
            let color_ref = vk::AttachmentReferenceBuilder::new()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let depth_ref = vk::AttachmentReferenceBuilder::new()
                .attachment(1)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            let resolve_ref = vk::AttachmentReferenceBuilder::new()
                .attachment(2)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let color_refs = [color_ref];
            let resolve_refs = [resolve_ref];
            let ats = [color_at, depth_at, resolve_at];
            let deps = [vk::SubpassDependencyBuilder::new()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                )];
            let subpass = vk::SubpassDescriptionBuilder::new()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_refs)
                .depth_stencil_attachment(&depth_ref)
                .resolve_attachments(&resolve_refs);
            let subpasses = [subpass];
            let rpci = vk::RenderPassCreateInfoBuilder::new()
                .attachments(&ats)
                .subpasses(&subpasses)
                .dependencies(&deps);
            unsafe { device.create_render_pass(&rpci, allocation_cbs()) }
                .expect("Could not create Vulkan renderpass")
        };

        let oneoff_cmd_pool = {
            let cpci = vk::CommandPoolCreateInfoBuilder::new()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(queues.get_primary_family());
            unsafe { device.create_command_pool(&cpci, allocation_cbs()) }
                .expect("Couldn't create one-off command pool")
        };

        let mut destroy_queues: Vec<VDODestroyQueue> = Default::default();
        destroy_queues.resize_with(INFLIGHT_FRAMES as usize, Default::default);

        (
            window,
            Self {
                entry,
                instance,
                ext_debug,
                surface,
                surface_format,
                physical,
                physical_limits,
                sample_count,
                device,
                queues: Arc::new(queues),
                vmalloc: Arc::new(Mutex::new(vmalloc)),
                mainpass,
                oneoff_cmd_pool: Arc::new(Mutex::new(oneoff_cmd_pool)),
                inflight_index: 0,
                destroy_queue: Arc::new(Mutex::new(Vec::default())),
                frame_destroy_queues: Arc::new(Mutex::new(destroy_queues)),
            },
        )
    }

    fn create_instance(
        entry: &EntryLoader,
        window: &Window,
        cfg: &Config,
    ) -> (Arc<InstanceLoader>, Option<DebugExts>) {
        let app_name = CString::new("BallX World").unwrap();
        let engine_name = CString::new("BallX World Engine").unwrap();

        let sdl_exts = window
            .vulkan_instance_extensions()
            .expect("Couldn't get a list of the required VK instance extensions");
        let avail_exts = unsafe {
            entry
                .enumerate_instance_extension_properties(None, None)
                .expect("Could not enumerate available Vulkan extensions")
        };
        let avail_enames: Vec<&CStr> = avail_exts
            .iter()
            .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
            .collect();
        let mut raw_exts: Vec<CString> = sdl_exts
            .into_iter()
            .map(|x| CString::new(x).expect("Invalid required VK instance extension"))
            .collect();
        for ext in raw_exts.iter() {
            let available = avail_enames.contains(&ext.as_c_str());
            if !available {
                panic!(
                    "Required Vulkan extension {} not present on this system! Update your drivers",
                    ext.to_str().unwrap()
                );
            }
        }

        let avail_layers = unsafe { entry.enumerate_instance_layer_properties(None) }
            .expect("Could not enumerate available Vulkan layers");
        let avail_lnames = avail_layers
            .iter()
            .map(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) });
        let mut raw_layers: Vec<CString> = Vec::new();

        let mut has_debug = false;
        if cfg.vk_debug_layers || cfg.dbg_renderdoc {
            let duname = unsafe { CStr::from_ptr(EXT_DEBUG_UTILS_EXTENSION_NAME) };
            if avail_enames.contains(&duname) {
                raw_exts.push(duname.to_owned());
                has_debug = true;
            }
            if !cfg.dbg_renderdoc {
                let lname = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
                if avail_lnames.clone().any(|e| e == lname.as_c_str()) {
                    raw_layers.push(lname);
                }
            }
        }

        let enabled_exts: Vec<*const c_char> = raw_exts.iter().map(|s| s.as_ptr()).collect();
        let enabled_layers: Vec<*const c_char> = raw_layers.iter().map(|s| s.as_ptr()).collect();

        let ai = vk::ApplicationInfoBuilder::new()
            .api_version(vk::make_api_version(0, 1, 2, 0))
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .application_name(&app_name)
            .engine_name(&engine_name);
        let ici = vk::InstanceCreateInfoBuilder::new()
            .application_info(&ai)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_exts);
        let instance = Arc::new(
            unsafe { InstanceLoaderBuilder::new().allocation_callbacks(allocation_cbs().unwrap()).build(entry, &ici) }
                .expect("Couldn't create Vulkan instance"),
        );

        if has_debug {
            let ext_debug = if !cfg.dbg_renderdoc {
                let mci = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
                    .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                    .pfn_user_callback(Some(debug_msg_callback));
                let msg =
                    unsafe { instance.create_debug_utils_messenger_ext(&mci, allocation_cbs()) }
                        .expect("Couldn't create debug messenger");
                log::debug!("Created Vulkan debug messenger");
                Some(DebugExts {
                    debug_messenger: msg,
                })
            } else {
                None
            };
            (instance, ext_debug)
        } else {
            (instance, None)
        }
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn load_shader_module(&self, path: &str) -> std::io::Result<vk::ShaderModule> {
        use std::fs;

        // read SPIR-V file
        let spv: Vec<u8> =
            fs::read(path).unwrap_or_else(|_| panic!("Could not read shader module from {}", path));

        // create module
        let smci = vk::ShaderModuleCreateInfo {
            p_code: spv.as_ptr() as *const u32,
            code_size: spv.len(),
            ..Default::default()
        };
        let sm = unsafe { self.device.create_shader_module(&smci, allocation_cbs()) }
            .result()
            .unwrap_or_else(|e| panic!("Could not create shader module from `{}`: {}", path, e));
        Ok(sm)
    }

    pub fn enqueue_destroy(&self, vdo: Box<dyn VulkanDeviceObject + Send>) {
        self.destroy_queue.lock().push(vdo);
    }

    pub fn get_destroy_queue_handle(&self) -> Weak<Mutex<VDODestroyQueue>> {
        Arc::downgrade(&self.destroy_queue)
    }

    pub fn flush_destroy_queue(&self) {
        let mut vmalloc = self.vmalloc.lock_traced("vmalloc", file!(), line!());
        for mut vdo in self
            .destroy_queue
            .lock_traced("vdo destroy queue", file!(), line!())
            .drain(..)
        {
            vdo.destroy(&mut vmalloc, self);
        }
        for queue in self
            .frame_destroy_queues
            .lock_traced("frame destroy queues", file!(), line!())
            .iter_mut()
        {
            for mut vdo in queue.drain(..) {
                vdo.destroy(&mut vmalloc, self);
            }
        }
    }

    pub fn destroy(self) {
        self.flush_destroy_queue();
        let Self {
            instance,
            ext_debug,
            surface,
            device,
            vmalloc,
            mainpass,
            oneoff_cmd_pool,
            ..
        } = self;
        if mainpass != vk::RenderPass::null() {
            unsafe {
                device.destroy_render_pass(mainpass, allocation_cbs());
            }
        }
        let mut vmalloc = Arc::try_unwrap(vmalloc)
            .unwrap_or_else(|_| panic!("Multiple references to vmalloc"))
            .into_inner();

        vmalloc.destroy();
        unsafe {
            instance.destroy_surface_khr(surface, allocation_cbs());
        }
        let oneoff_cmd_pool = Arc::try_unwrap(oneoff_cmd_pool)
            .unwrap_or_else(|_| panic!("Multiple references to oneoff_cmd_pool"))
            .into_inner();
        unsafe {
            device.destroy_command_pool(oneoff_cmd_pool, allocation_cbs());
        }
        unsafe {
            device.destroy_device(allocation_cbs());
        }
        if let Some(ext_debug) = ext_debug {
            unsafe {
                instance.destroy_debug_utils_messenger_ext(
                    ext_debug.debug_messenger,
                    allocation_cbs(),
                );
            }
        }
        unsafe {
            instance.destroy_instance(allocation_cbs());
        }
    }
}

impl Swapchain {
    pub fn new(window: &Window, handles: &RenderingHandles, cfg: &Config) -> Self {
        let mut sch = Self {
            swapchain: vk::SwapchainKHR::null(),
            swapimage_size: Default::default(),
            swapimages: Default::default(),
            swapimageviews: Vec::new(),
            depth_image: OwnedImage::new(),
            color_image: OwnedImage::new(),
            inflight_render_finished_semaphores: Vec::new(),
            inflight_image_available_semaphores: Vec::new(),
            inflight_fences: Vec::new(),
            outdated: false,
            dynamic_state: DynamicState::default(),
            framebuffers: Vec::new(),
        };
        sch.recreate_swapchain(window, handles, cfg);
        sch
    }

    fn destroy_vk_objs(&mut self, handles: &RenderingHandles, destroy_swapchain: bool) {
        unsafe {
            handles.device.device_wait_idle().unwrap();
        }
        for fence in self.inflight_fences.drain(..) {
            if fence != vk::Fence::null() {
                unsafe {
                    handles.device.destroy_fence(fence, allocation_cbs());
                }
            }
        }
        for semaphore in self
            .inflight_render_finished_semaphores
            .drain(..)
            .chain(self.inflight_image_available_semaphores.drain(..))
        {
            if !semaphore.is_null() {
                unsafe {
                    handles
                        .device
                        .destroy_semaphore(semaphore, allocation_cbs());
                }
            }
        }
        for fb in self.framebuffers.drain(..) {
            if !fb.is_null() {
                unsafe {
                    handles
                        .device
                        .destroy_framebuffer(fb, allocation_cbs());
                }
            }
        }
        for iv in self.swapimageviews.drain(..) {
            if !iv.is_null() {
                unsafe {
                    handles
                        .device
                        .destroy_image_view(iv, allocation_cbs());
                }
            }
        }
        // Don't destroy swapimages - they are owned by the swapchain object
        self.depth_image.destroy(
            &mut handles.vmalloc.lock_traced("vmalloc", file!(), line!()),
            handles,
        );
        self.color_image.destroy(
            &mut handles.vmalloc.lock_traced("vmalloc", file!(), line!()),
            handles,
        );
        if destroy_swapchain && !self.swapchain.is_null() {
            unsafe {
                handles
                    .device
                    .destroy_swapchain_khr(self.swapchain, allocation_cbs());
            }
            self.swapchain = vk::SwapchainKHR::null();
        }
    }

    fn recreate_swapchain(&mut self, window: &Window, handles: &RenderingHandles, cfg: &Config) {
        self.destroy_vk_objs(handles, false);
        let dimensions = window.vulkan_drawable_size();

        let caps = unsafe {
            handles
                .instance
                .get_physical_device_surface_capabilities_khr(handles.physical, handles.surface)
        }
        .expect("Failed to get surface capabilities");

        let present_modes = unsafe {
            handles
                .instance
                .get_physical_device_surface_present_modes_khr(
                    handles.physical,
                    handles.surface,
                    None,
                )
        }
        .expect("Failed to get surface present modes");

        let present_mode: vk::PresentModeKHR = {
            if !cfg.render_wait_for_vsync {
                present_modes
                    .iter()
                    .copied()
                    .find(|p| *p == vk::PresentModeKHR::MAILBOX_KHR)
                    .or_else(|| {
                        present_modes
                            .iter()
                            .copied()
                            .find(|p| *p == vk::PresentModeKHR::IMMEDIATE_KHR)
                    })
                    .unwrap_or(vk::PresentModeKHR::FIFO_KHR)
            } else {
                vk::PresentModeKHR::FIFO_KHR
            }
        };

        let extent = vk::Extent2D {
            width: clamp(
                dimensions.0,
                caps.min_image_extent.width,
                caps.max_image_extent.width,
            ),
            height: clamp(
                dimensions.1,
                caps.min_image_extent.height,
                caps.max_image_extent.height,
            ),
        };
        self.swapimage_size = extent;

        let image_count = u32::min(caps.min_image_count + 1, caps.max_image_count);

        let (image_sharing_mode, queue_family_indices) = (
            vk::SharingMode::EXCLUSIVE,
            &[handles.queues.get_primary_family()],
        );

        if cfg.debug_logging {
            log::debug!(
                "Recreating swapchain with size ({w}, {h}), {ic} images, {pm:?} present mode",
                w = extent.width,
                h = extent.height,
                ic = image_count,
                pm = present_mode
            );
        }

        let sci = vk::SwapchainCreateInfoKHRBuilder::new()
            .surface(handles.surface)
            .min_image_count(image_count)
            .image_color_space(handles.surface_format.color_space)
            .image_format(handles.surface_format.format)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(queue_family_indices)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(self.swapchain)
            .image_array_layers(1);

        let old_swapchain = self.swapchain;
        self.swapchain = unsafe { handles.device.create_swapchain_khr(&sci, allocation_cbs()) }
            .expect("Failed to create swapchain");
        unsafe {
            handles
                .device
                .destroy_swapchain_khr(old_swapchain, allocation_cbs());
        }

        self.outdated = false;

        self.swapimages = unsafe {
            handles
                .device
                .get_swapchain_images_khr(self.swapchain, None)
                .expect("Failed to get swapchain images")
        };
        let image_count = self.swapimages.len() as u32;

        self.swapimageviews = {
            let mut iv = Vec::with_capacity(image_count as usize);
            for swi in self.swapimages.iter() {
                let ivci = vk::ImageViewCreateInfoBuilder::new()
                    .view_type(vk::ImageViewType::_2D)
                    .format(handles.surface_format.format)
                    .components(identity_components())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(*swi);
                iv.push(
                    unsafe { handles.device.create_image_view(&ivci, allocation_cbs()) }
                        .expect("Failed to create swapchain imageview"),
                );
            }
            iv
        };

        self.depth_image = {
            let qfis = [handles.queues.get_primary_family()];
            let ici = vk::ImageCreateInfoBuilder::new()
                .image_type(vk::ImageType::_2D)
                .format(vk::Format::D32_SFLOAT_S8_UINT)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(handles.sample_count)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&qfis)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let aci = vma::AllocationCreateInfo {
                usage: vma::MemoryUsage::GpuOnly,
                flags: vma::AllocationCreateFlags::DEDICATED_MEMORY,
                ..Default::default()
            };
            OwnedImage::from(
                &mut handles.vmalloc.lock_traced("vmalloc", file!(), line!()),
                handles,
                &ici,
                &aci,
                vk::ImageViewType::_2D,
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            )
        };
        self.depth_image
            .give_name(handles, || "swapchain.depth_image");

        self.color_image = {
            let qfis = [handles.queues.get_primary_family()];
            let ici = vk::ImageCreateInfoBuilder::new()
                .image_type(vk::ImageType::_2D)
                .format(handles.surface_format.format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(handles.sample_count)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&qfis)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let aci = vma::AllocationCreateInfo {
                usage: vma::MemoryUsage::GpuOnly,
                flags: vma::AllocationCreateFlags::DEDICATED_MEMORY,
                ..Default::default()
            };
            OwnedImage::from(
                &mut handles.vmalloc.lock_traced("vmalloc", file!(), line!()),
                handles,
                &ici,
                &aci,
                vk::ImageViewType::_2D,
                vk::ImageAspectFlags::COLOR,
            )
        };
        self.depth_image
            .give_name(handles, || "swapchain.depth_image");

        for img in self.swapimageviews.iter() {
            let attachs = [
                self.color_image.image_view,
                self.depth_image.image_view,
                *img,
            ];
            let fci = vk::FramebufferCreateInfoBuilder::new()
                .render_pass(handles.mainpass)
                .attachments(&attachs)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            self.framebuffers.push(
                unsafe { handles.device.create_framebuffer(&fci, allocation_cbs()) }
                    .expect("Failed to create framebuffer"),
            );
        }

        for _ in 0..INFLIGHT_FRAMES {
            let sci = vk::SemaphoreCreateInfoBuilder::new().build_dangling();
            let rfsem = unsafe { handles.device.create_semaphore(&sci, allocation_cbs()) }
                .expect("Could not create Vulkan semaphore");
            let iasem = unsafe { handles.device.create_semaphore(&sci, allocation_cbs()) }
                .expect("Could not create Vulkan semaphore");
            self.inflight_render_finished_semaphores.push(rfsem);
            self.inflight_image_available_semaphores.push(iasem);

            let fci = vk::FenceCreateInfoBuilder::new()
                .flags(vk::FenceCreateFlags::SIGNALED);
            let iff = unsafe { handles.device.create_fence(&fci, allocation_cbs()) }
                .expect("Could not create Vulkan fence");
            self.inflight_fences.push(iff);
        }

        self.dynamic_state.viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
    }
}

impl RenderingContext {
    pub fn new(sdl_video: &sdl2::VideoSubsystem, cfg: &Config) -> RenderingContext {
        let (window, handles) = RenderingHandles::new(sdl_video, cfg);
        let swapchain = Swapchain::new(&window, &handles, cfg);

        let pipeline_cache = {
            let pci = vk::PipelineCacheCreateInfoBuilder::new();
            unsafe { handles.device.create_pipeline_cache(&pci, allocation_cbs()) }
                .expect("Couldn't create pipeline cache")
        };

        let cmd_pool = {
            let cpci = vk::CommandPoolCreateInfoBuilder::new()
                .queue_family_index(handles.queues.get_primary_family())
                .flags(
                    vk::CommandPoolCreateFlags::TRANSIENT
                        | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                );
            unsafe { handles.device.create_command_pool(&cpci, allocation_cbs()) }
                .expect("Couldn't create main command pool")
        };

        let inflight_cmds = {
            let cbai = vk::CommandBufferAllocateInfoBuilder::new()
                .command_buffer_count(INFLIGHT_FRAMES)
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY);
            unsafe { handles.device.allocate_command_buffers(&cbai) }
                .expect("Couldn't allocate command buffers")
        };
        assert_eq!(inflight_cmds.len(), INFLIGHT_FRAMES as usize);

        RenderingContext {
            window,
            handles,
            swapchain: ManuallyDrop::new(swapchain),
            frame_index: 0,
            cmd_pool,
            inflight_cmds,
            pipeline_cache,
        }
    }

    pub fn destroy(mut self) {
        if self.handles.device.handle.is_null() {
            return;
        }
        unsafe {
            self.handles.device.device_wait_idle().unwrap();
        }
        self.handles.flush_destroy_queue();
        if !self.pipeline_cache.is_null() {
            unsafe {
                self.handles
                    .device
                    .destroy_pipeline_cache(self.pipeline_cache, allocation_cbs());
            }
            self.pipeline_cache = vk::PipelineCache::null();
        }
        self.inflight_cmds.clear();
        if !self.cmd_pool.is_null() {
            unsafe {
                self.handles
                    .device
                    .destroy_command_pool(self.cmd_pool, allocation_cbs());
            }
            self.cmd_pool = vk::CommandPool::null();
        }
        unsafe {
            self.swapchain.destroy_vk_objs(&self.handles, true);
            ManuallyDrop::drop(&mut self.swapchain);
        }
        self.handles.destroy();
    }

    /// Returns None if e.g. swapchain is in the process of being recreated
    #[allow(clippy::modulo_one)] // There may only be one inflight frame configured
    pub fn frame_begin_prepass<'cw>(
        &mut self,
        cfg: &Config,
        client_world: &'cw ClientWorld,
        delta_time: f64,
    ) -> Option<PrePassFrameContext<'_, 'cw>> {
        let device = &self.handles.device;
        let old_inflight_index = self.handles.inflight_index;
        self.handles.inflight_index = (self.handles.inflight_index + 1) % INFLIGHT_FRAMES;
        let inflight_index = self.handles.inflight_index as usize;
        let inflight_fence = self.swapchain.inflight_fences[inflight_index];
        {
            let _p_zone = bxw_util::tracy_client::Span::new(
                "Wait for inflight fences",
                "mainloop",
                file!(),
                line!(),
                4,
            );
            unsafe { device.wait_for_fences(&[inflight_fence], true, u64::max_value()) }
                .expect("Failed waiting for fence");
            unsafe { device.reset_fences(&[inflight_fence]) }.expect("Couldn't reset fence");
        }

        let mut vmalloc = self
            .handles
            .vmalloc
            .lock_traced("vmalloc", file!(), line!());

        let memstats = vmalloc.calculate_stats();
        if let Ok(memstats) = memstats {
            DEBUG_DATA
                .gpu_usage_bytes
                .store(memstats.total.usedBytes as i64, Ordering::Release);
        }

        {
            let _p_zone = bxw_util::tracy_client::Span::new(
                "Run frame destroy queue tasks",
                "mainloop",
                file!(),
                line!(),
                4,
            );
            for mut vdo in
                self.handles.frame_destroy_queues.lock()[old_inflight_index as usize].drain(..)
            {
                vdo.destroy(&mut vmalloc, &self.handles);
            }
        }
        vmalloc.set_current_frame_index(self.frame_index);
        drop(vmalloc);
        let mut dq = self.handles.destroy_queue.lock();
        std::mem::swap(
            &mut self.handles.frame_destroy_queues.lock()[old_inflight_index as usize],
            &mut dq,
        );
        dq.clear();
        drop(dq);
        self.frame_index = self.frame_index.wrapping_add(1);

        let dims = {
            let d = self.window.vulkan_drawable_size();
            [max(16, d.0), max(16, d.1)]
        };

        if self.swapchain.outdated {
            self.swapchain
                .recreate_swapchain(&self.window, &self.handles, cfg);
            return None;
        }

        let image_index = {
            let result = unsafe {
                self.handles.device.acquire_next_image_khr(
                    self.swapchain.swapchain,
                    u64::MAX,
                    self.swapchain.inflight_image_available_semaphores[inflight_index],
                    vk::Fence::null(),
                )
            };
            if result.raw == vk::Result::SUBOPTIMAL_KHR {
                self.swapchain.outdated = true;
            }
            if result.raw == vk::Result::ERROR_OUT_OF_DATE_KHR {
                self.swapchain.outdated = true;
                self.swapchain
                    .recreate_swapchain(&self.window, &self.handles, cfg);
                return None;
            } else if let Some(idx) = result.value {
                idx
            } else {
                panic!("{:?}", result.raw)
            }
        };

        let cmd = {
            let buf = self.inflight_cmds[inflight_index];
            unsafe { device.reset_command_buffer(buf, vk::CommandBufferResetFlags::empty()) }
                .expect("Couldn't reset frame cmd buffer");
            let cbbi = vk::CommandBufferBeginInfoBuilder::new()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { device.begin_command_buffer(buf, &cbbi) }
                .expect("Couldn't start recording command buffer");
            buf
        };

        Some(PrePassFrameContext {
            rctx: self,
            cmd,
            client_world,
            delta_time,
            dims,
            image_index: image_index as usize,
            inflight_index,
            _phantom: PhantomData,
        })
    }

    pub fn frame_goto_pass<'r, 'cw>(
        fctx: PrePassFrameContext<'r, 'cw>,
    ) -> InPassFrameContext<'r, 'cw> {
        let PrePassFrameContext {
            rctx: me,
            cmd,
            client_world,
            delta_time,
            dims,
            image_index,
            inflight_index,
            _phantom,
        } = fctx;

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.1, 0.1, 0.1, 1.0],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: me.swapchain.dynamic_state.viewport.width as u32,
                height: me.swapchain.dynamic_state.viewport.height as u32,
            },
        };
        let clears = [clear_color, clear_depth];
        let rpbi = vk::RenderPassBeginInfoBuilder::new()
            .render_pass(me.handles.mainpass)
            .framebuffer(me.swapchain.framebuffers[image_index])
            .render_area(area)
            .clear_values(&clears);
        unsafe {
            me.handles
                .device
                .cmd_begin_render_pass(cmd, &rpbi, vk::SubpassContents::INLINE);
        }

        InPassFrameContext {
            rctx: me,
            cmd,
            client_world,
            delta_time,
            dims,
            image_index,
            inflight_index,
            _phantom: PhantomData,
        }
    }

    pub fn frame_goto_postpass<'r, 'cw>(
        fctx: InPassFrameContext<'r, 'cw>,
    ) -> PostPassFrameContext<'r, 'cw> {
        let InPassFrameContext {
            rctx: me,
            cmd,
            client_world,
            delta_time,
            dims,
            image_index,
            inflight_index,
            _phantom,
        } = fctx;

        unsafe {
            me.handles.device.cmd_end_render_pass(cmd);
        }

        PostPassFrameContext {
            rctx: me,
            cmd,
            client_world,
            delta_time,
            dims,
            image_index,
            inflight_index,
            _phantom: PhantomData,
        }
    }

    pub fn frame_finish(fctx: PostPassFrameContext) {
        let PostPassFrameContext {
            rctx: me,
            cmd,
            image_index,
            inflight_index,
            ..
        } = fctx;

        unsafe { me.handles.device.end_command_buffer(cmd) }
            .expect("Couldn't end recording command buffer");

        let imgavail = [me.swapchain.inflight_image_available_semaphores[inflight_index]];
        let rendfinish = [me.swapchain.inflight_render_finished_semaphores[inflight_index]];
        let cmds = [cmd];
        let wss = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let si = vk::SubmitInfoBuilder::new()
            .wait_semaphores(&imgavail)
            .signal_semaphores(&rendfinish)
            .command_buffers(&cmds)
            .wait_dst_stage_mask(&wss);

        let queue = me.handles.queues.lock_primary_queue();

        bxw_util::tracy_client::message("vk Frame Queue Submit", 4);
        unsafe {
            me.handles.device.queue_submit(
                *queue,
                &[si],
                me.swapchain.inflight_fences[inflight_index],
            )
        }
        .expect("Couldn't submit frame command buffer");

        let swchs = [me.swapchain.swapchain];
        let imgids = [image_index as u32];
        let pi = vk::PresentInfoKHRBuilder::new()
            .wait_semaphores(&rendfinish)
            .swapchains(&swchs)
            .image_indices(&imgids);
        bxw_util::tracy_client::message("vk Queue Present", 4);
        let result = unsafe { me.handles.device.queue_present_khr(*queue, &pi) };
        if result.raw == vk::Result::SUBOPTIMAL_KHR
            || result.raw == vk::Result::ERROR_OUT_OF_DATE_KHR
        {
            me.swapchain.outdated = true;
        } else if result.is_err() {
            panic!("{:?}", result.raw);
        }
        bxw_util::tracy_client::message("Presented", 4);
    }
}
