use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub particle_size: f32,
    pub camera_movement_speed: f32,
    pub camera_smoothness: f32,
    pub zoom_step_factor: f32,
    pub start_in_fullscreen: bool,
    pub show_cursor: bool,
    pub cursor_size: f32,
    pub brush_power: i32,
    pub auto_dt: bool,
    pub dt: f64,
    pub palette: String,
    pub shader: String,
    pub position_setter: String,
    pub cursor_action_left: String,
    pub cursor_action_right: String,
    pub matrix_gui_step_size: f32,
    pub keep_particle_size_independent_of_zoom: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            particle_size: 0.5,
            camera_movement_speed: 0.2,
            camera_smoothness: 0.1,
            zoom_step_factor: 1.1,
            start_in_fullscreen: false,
            show_cursor: true,
            cursor_size: 0.05,
            brush_power: 10,
            auto_dt: true,
            dt: 0.016,
            palette: "Natural Rainbow".to_string(),
            shader: "default".to_string(),
            position_setter: "Random".to_string(),
            cursor_action_left: "Move".to_string(),
            cursor_action_right: "Delete".to_string(),
            matrix_gui_step_size: 0.1,
            keep_particle_size_independent_of_zoom: true,
        }
    }
}

impl AppSettings {
    const SETTINGS_FILE: &'static str = "settings.toml";

    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        if Path::new(Self::SETTINGS_FILE).exists() {
            let contents = fs::read_to_string(Self::SETTINGS_FILE)?;
            let settings: AppSettings = toml::from_str(&contents)?;
            Ok(settings)
        } else {
            Ok(Self::default())
        }
    }
}
