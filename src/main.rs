#![allow(clippy::upper_case_acronyms)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unused_must_use)]

pub mod client;
pub mod config;
pub mod network;
pub mod server;
pub mod util;
pub mod vk;

use bxw_util::debug_data::TrackingAllocator;
use bxw_util::log;

#[global_allocator]
static TRACK_ALLOCATOR: TrackingAllocator<rpmalloc::RpMalloc> = TrackingAllocator {
    allocator: rpmalloc::RpMalloc,
};

fn main() {
    setup_logging();
    bxw_util::tracy_client::Client::start();
    bxw_util::tracy_client::set_thread_name!("bxw-main");
    bxw_util::sodiumoxide::init().expect("Couldn't initialize cryptography library");
    if std::env::args().any(|a| a == "-server") {
        server::main::server_main();
    } else {
        client::main::client_main();
    }
}

fn setup_logging() {
    log4rs::init_file("settings_logging.toml", Default::default())
        .expect("Couldn't initialize the logging subsystem");
    log::info!("Logging setup complete");
}

mod rpmalloc {
    use rpmalloc_sys as rpm;
    use rpmalloc_sys::c_void;
    use std::alloc::GlobalAlloc;
    use std::alloc::Layout;

    pub struct RpMalloc;

    unsafe impl GlobalAlloc for RpMalloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            unsafe { rpm::rpaligned_alloc(layout.align().max(16), layout.size()) as *mut u8 }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
            unsafe {
                rpm::rpfree(ptr as *mut c_void);
            }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            unsafe {
                rpm::rpaligned_realloc(
                    ptr as *mut c_void,
                    layout.align().max(16),
                    new_size,
                    layout.size(),
                    0,
                ) as *mut u8
            }
        }
    }
}
