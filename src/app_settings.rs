use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Application settings that control various aspects of the particle simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    /// Size of particles in the simulation
    pub particle_size: f32,
    /// Speed at which the camera moves when controlled
    pub camera_movement_speed: f32,
    /// Smoothness factor for camera movement (lower values = smoother)
    pub camera_smoothness: f32,
    /// Factor by which zoom level changes per step
    pub zoom_step_factor: f32,
    /// Size of the cursor in the simulation
    pub cursor_size: f32,
    /// Time step (delta time) for the simulation in seconds
    pub time_step: f64,
    /// Name of the color palette to use
    pub palette: String,
    /// Method used to set initial particle positions
    pub position_setter: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            particle_size: 0.5,
            camera_movement_speed: 0.2,
            camera_smoothness: 0.1,
            zoom_step_factor: 1.1,
            cursor_size: 0.5,
            time_step: 0.016,
            palette: "Natural Rainbow".to_string(),
            position_setter: "Random".to_string(),
        }
    }
}

impl AppSettings {
    const SETTINGS_FILE: &'static str = "settings.toml";

    /// Loads settings from the settings file, or returns default settings if the file doesn't exist
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
