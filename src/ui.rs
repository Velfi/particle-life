//! UI components and timing systems for the particle life simulator
//! 
//! This module provides:
//! - Selection management for UI components
//! - Frame timing and FPS tracking
//! - Simulation loop control
//! 
//! The UI system is designed to be modular and extensible, allowing for
//! easy addition of new UI components and settings.

/// Manages a collection of named items with an active selection
/// 
/// Used for:
/// - Color palettes
/// - Position patterns
/// - Type distributions
/// - Matrix generators
pub struct SelectionManager<T> {
    /// List of named items
    items: Vec<NamedItem<T>>,
    /// Index of the currently active item
    active_index: usize,
}

/// A named item in a selection manager
pub struct NamedItem<T> {
    /// Display name of the item
    pub name: String,
    /// The actual item data
    pub object: T,
}

impl<T> SelectionManager<T> {
    /// Creates a new selection manager with the given items
    /// 
    /// Parameters:
    /// - items: Vector of (name, object) pairs
    pub fn new(items: Vec<(String, T)>) -> Self {
        let items = items
            .into_iter()
            .map(|(name, object)| NamedItem { name, object })
            .collect();
        Self {
            items,
            active_index: 0,
        }
    }

    /// Returns a reference to the currently active item
    pub fn get_active(&self) -> &T {
        &self.items[self.active_index].object
    }

    /// Returns the name of the currently active item
    pub fn get_active_name(&self) -> &str {
        &self.items[self.active_index].name
    }

    /// Returns the index of the currently active item
    pub fn get_active_index(&self) -> usize {
        self.active_index
    }

    /// Returns a vector of all item names
    pub fn get_item_names(&self) -> Vec<&str> {
        self.items.iter().map(|item| item.name.as_str()).collect()
    }

    /// Sets the active item by index
    /// 
    /// Parameters:
    /// - index: Index of the item to activate
    pub fn set_active(&mut self, index: usize) {
        if index < self.items.len() {
            self.active_index = index;
        }
    }
}

/// Frame timing and FPS tracking system
/// 
/// Features:
/// - Frame time history
/// - Average FPS calculation
/// - Target FPS tracking
pub struct Clock {
    /// Time of the last frame
    last_time: std::time::Instant,
    /// History of frame times for averaging
    dt_history: Vec<f64>,
    /// Target frames per second
    target_fps: f64,
}

impl Clock {
    /// Creates a new clock with the specified target FPS
    /// 
    /// Parameters:
    /// - target_fps: Target frames per second
    pub fn new(target_fps: f64) -> Self {
        Self {
            last_time: std::time::Instant::now(),
            dt_history: Vec::with_capacity(60),
            target_fps,
        }
    }

    /// Updates the clock with the current frame time
    /// 
    /// Maintains a rolling history of the last 60 frames
    pub fn tick(&mut self) {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_time).as_secs_f64();
        self.last_time = now;

        self.dt_history.push(dt);
        if self.dt_history.len() > 60 {
            self.dt_history.remove(0);
        }
    }

    /// Returns the last frame time in milliseconds
    pub fn get_dt_millis(&self) -> f64 {
        self.dt_history
            .last()
            .copied()
            .unwrap_or(1.0 / self.target_fps)
            * 1000.0
    }

    /// Returns the average frames per second
    /// 
    /// Calculated from the frame time history
    pub fn get_avg_framerate(&self) -> f64 {
        if self.dt_history.is_empty() {
            return self.target_fps;
        }
        let avg_dt: f64 = self.dt_history.iter().sum::<f64>() / self.dt_history.len() as f64;
        if avg_dt > 0.0 {
            1.0 / avg_dt
        } else {
            self.target_fps
        }
    }
}

/// Controls the simulation loop state
pub struct Loop {
    /// Whether the simulation is paused
    pub pause: bool,
}

impl Loop {
    /// Creates a new loop controller
    pub fn new() -> Self {
        Self { pause: false }
    }
}
