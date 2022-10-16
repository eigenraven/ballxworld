use bxw_util::parking_lot::RwLock;
use bxw_util::*;
use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;
use std::io::prelude::*;
use std::net::{SocketAddr, SocketAddrV4};
use std::sync::Arc;
use toml;

#[derive(Clone, Debug, Serialize, Deserialize, SmartDefault)]
#[serde(default)]
pub struct Window {
    #[default = 1280]
    pub width: u32,
    #[default = 720]
    pub height: u32,
    #[default = false]
    pub fullscreen: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, SmartDefault)]
#[serde(default)]
pub struct Render {
    #[default = 4]
    pub samples: u32,
    #[default = true]
    pub wait_for_vsync: bool,
    #[default(None)]
    pub fps_lock: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, SmartDefault)]
#[serde(default)]
pub struct Performance {
    #[default = 10]
    pub draw_distance: u32,
    #[default = 10]
    pub update_distance: u32,
    // reserve two cores for the base process and any other threaded activity
    #[default(bxw_util::num_cpus::get().saturating_sub(2).max(1) as u32)]
    pub threads: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, SmartDefault)]
#[serde(default)]
pub struct Network {
    #[default(vec![SocketAddr::V4(SocketAddrV4::new(
            std::net::Ipv4Addr::new(0, 0, 0, 0),
            20138,
    ))])]
    pub listen_addresses: Vec<SocketAddr>,
}

#[derive(Clone, Debug, Serialize, Deserialize, SmartDefault)]
#[serde(default)]
pub struct Debugging {
    pub logging: bool,
    pub vk_debug_layers: bool,
    #[serde(skip)]
    pub renderdoc: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub window: Window,
    pub render: Render,
    pub performance: Performance,
    pub network: Network,
    pub debugging: Debugging,
}

pub type ConfigHandle = Arc<RwLock<Config>>;

impl Config {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn standard_load() -> ConfigHandle {
        let mut cfg = Config::new();
        let cfg_file = std::fs::File::open("settings.toml");
        match cfg_file {
            Err(_) => {
                log::warn!("Creating new settings.toml");
            }
            Ok(mut cfg_file) => {
                let mut cfg_text = String::new();
                cfg_file
                    .read_to_string(&mut cfg_text)
                    .expect("Error reading settings.toml");
                cfg.load_from_toml(&cfg_text);
            }
        }
        let cfg_file = std::fs::File::create("settings.toml");
        let cfg_text = cfg.save_toml();
        cfg_file
            .expect("Couldn't open settings.toml for writing")
            .write_all(cfg_text.as_bytes())
            .expect("Couldn't write to settings.toml");
        Arc::new(RwLock::new(cfg))
    }

    pub fn load_from_toml(&mut self, config: &str) {
        *self = toml::from_str(config).expect("Couldn't load config from TOML");
        /*
        self.window_width = toml_doc["window"]["width"]
            .as_integer()
            .map_or(self.window_width, |v| v as u32);
        self.window_height = toml_doc["window"]["height"]
            .as_integer()
            .map_or(self.window_height, |v| v as u32);
        self.window_fullscreen = toml_doc["window"]["width"]
            .as_bool()
            .unwrap_or(self.window_fullscreen);
        self.window_monitor = toml_doc["window"]["monitor"]
            .as_integer()
            .map_or(self.window_monitor, |v| v as u32);

        self.render_samples = toml_doc["render"]["samples"]
            .as_integer()
            .map_or(self.render_samples, |v| v as u32);
        self.render_wait_for_vsync = toml_doc["render"]["wait_for_vsync"]
            .as_bool()
            .unwrap_or(self.render_wait_for_vsync);
        self.render_fps_lock =
            toml_doc["render"]["fps_lock"]
                .as_integer()
                .map_or(self.render_fps_lock, |v| {
                    if v == 0 {
                        None
                    } else {
                        Some(v as u32)
                    }
                });

        self.performance_load_distance = toml_doc["performance"]["load_distance"]
            .as_integer()
            .map_or(self.performance_load_distance, |v| v as u32);
        self.performance_draw_distance = toml_doc["performance"]["draw_distance"]
            .as_integer()
            .map_or(self.performance_draw_distance, |v| v as u32);
        self.performance_threads = toml_doc["performance"]["threads"]
            .as_integer()
            .map_or(self.performance_threads, |v| v as u32);

        self.server_listen_addresses = toml_doc["server"]["listen_addresses"]
            .as_array()
            .map_or(std::mem::take(&mut self.server_listen_addresses), |varr| {
                varr.iter().map(|v| v.as_str()
                    .expect("Address not a IP:Port string in config server.listen_addresses [type is not a string]")
                    .parse::<SocketAddr>()
                    .expect("Address not a valid IP:Port string in config server.listen_addresses [can't parse]")).
                    collect_vec()
            });
        self.server_mtu = toml_doc["server"]["mtu"]
            .as_integer()
            .map_or(self.server_mtu, |v| v as u16)
            .max(1000)
            .min(9216);

        self.debug_logging = toml_doc["debug"]["enable_logging"]
            .as_bool()
            .unwrap_or(self.debug_logging);
        self.vk_debug_layers = toml_doc["debug"]["enable_vk_layers"]
            .as_bool()
            .unwrap_or(self.vk_debug_layers);

        self.toml_doc = Some(toml_doc);
        */
    }

    #[allow(clippy::cast_lossless)]
    pub fn save_toml(&mut self) -> String {
        toml::to_string_pretty(self).expect("Couldn't serialize config")
    }
}
