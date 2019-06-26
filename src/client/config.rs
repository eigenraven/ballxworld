#[derive(Clone, Debug)]
pub struct Config {
    pub window_width: u32,
    pub window_height: u32,
    pub window_fullscreen: bool,
    pub window_monitor: u32,

    pub render_wait_for_vsync: bool,
    pub render_fps_lock: Option<u32>,

    toml_doc: Option<toml_edit::Document>,
}

impl Config {
    pub fn new() -> Self {
        Self {
            window_width: 1280,
            window_height: 720,
            window_fullscreen: false,
            window_monitor: 0,

            render_wait_for_vsync: false,
            render_fps_lock: None,

            toml_doc: None,
        }
    }

    pub fn load_from_toml(&mut self, config: &str) {
        use toml_edit::*;
        let toml_doc: Document = config
            .parse::<Document>()
            .expect("Invalid configuration TOML.");

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

        self.toml_doc = Some(toml_doc);
    }

    #[allow(clippy::cast_lossless)]
    pub fn save_toml(&mut self) -> String {
        use toml_edit::*;
        let mut toml_doc = std::mem::replace(&mut self.toml_doc, None).unwrap_or_default();

        toml_doc["window"]["width"] = Item::Value(Value::from(self.window_width as i64));
        toml_doc["window"]["height"] = Item::Value(Value::from(self.window_height as i64));
        toml_doc["window"]["fullscreen"] = Item::Value(Value::from(self.window_fullscreen));
        toml_doc["window"]["monitor"] = Item::Value(Value::from(self.window_monitor as i64));

        toml_doc["render"]["wait_for_vsync"] = Item::Value(Value::from(self.render_wait_for_vsync));
        toml_doc["render"]["fps_lock"] =
            Item::Value(Value::from(self.render_fps_lock.unwrap_or(0) as i64));

        self.toml_doc = Some(toml_doc);
        self.toml_doc.as_ref().unwrap().to_string()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}
