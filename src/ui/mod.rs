pub struct SelectionManager<T> {
    items: Vec<NamedItem<T>>,
    active_index: usize,
}

pub struct NamedItem<T> {
    pub object: T,
}

impl<T> SelectionManager<T> {
    pub fn new(items: Vec<(String, T)>) -> Self {
        let items = items
            .into_iter()
            .map(|(_, object)| NamedItem { object })
            .collect();
        Self {
            items,
            active_index: 0,
        }
    }

    pub fn get_active(&self) -> &T {
        &self.items[self.active_index].object
    }
}

pub struct Clock {
    last_time: std::time::Instant,
    dt_history: Vec<f64>,
    target_fps: f64,
}

impl Clock {
    pub fn new(target_fps: f64) -> Self {
        Self {
            last_time: std::time::Instant::now(),
            dt_history: Vec::with_capacity(60),
            target_fps,
        }
    }

    pub fn tick(&mut self) {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_time).as_secs_f64();
        self.last_time = now;

        self.dt_history.push(dt);
        if self.dt_history.len() > 60 {
            self.dt_history.remove(0);
        }
    }

    pub fn get_dt_millis(&self) -> f64 {
        self.dt_history
            .last()
            .copied()
            .unwrap_or(1.0 / self.target_fps)
            * 1000.0
    }

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

pub struct Loop {
    pub pause: bool,
}

impl Loop {
    pub fn new() -> Self {
        Self { pause: false }
    }
}
