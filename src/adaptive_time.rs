// Adaptive time stepping system for particle life simulation
use crate::soa_physics::ParticleArrays;

/// Adaptive time stepping controller
pub struct AdaptiveTimeController {
    /// Target frame rate (frames per second)
    pub target_fps: f64,
    
    /// Base time step
    pub base_dt: f64,
    
    /// Current adaptive time step
    pub current_dt: f64,
    
    /// Maximum allowed time step
    pub max_dt: f64,
    
    /// Minimum allowed time step  
    pub min_dt: f64,
    
    /// Performance metrics
    pub frame_times: Vec<f64>,
    pub avg_frame_time: f64,
    
    /// Particle density metrics
    pub high_density_threshold: f64,
    pub low_density_threshold: f64,
    
    /// Adaptation parameters
    pub adaptation_rate: f64,
    pub stability_buffer: f64,
}

impl AdaptiveTimeController {
    pub fn new(target_fps: f64, base_dt: f64) -> Self {
        Self {
            target_fps,
            base_dt,
            current_dt: base_dt,
            max_dt: base_dt * 2.0,
            min_dt: base_dt * 0.25,
            frame_times: Vec::with_capacity(60),
            avg_frame_time: 1.0 / target_fps,
            high_density_threshold: 0.8,  // High density when 80% of particles are active
            low_density_threshold: 0.3,   // Low density when 30% of particles are active
            adaptation_rate: 0.1,          // How quickly to adapt (10% change per frame)
            stability_buffer: 0.2,         // 20% buffer to prevent oscillation
        }
    }
    
    /// Update the adaptive time step based on performance and particle density
    pub fn update(&mut self, frame_time: f64, particles: &ParticleArrays) {
        // Add frame time to history
        self.frame_times.push(frame_time);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
        
        // Calculate average frame time
        self.avg_frame_time = self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64;
        
        // Calculate performance metrics
        let target_frame_time = 1.0 / self.target_fps;
        let performance_ratio = self.avg_frame_time / target_frame_time;
        
        // Calculate particle density metrics
        let density_metrics = self.calculate_density_metrics(particles);
        
        // Determine optimal time step
        let optimal_dt = self.calculate_optimal_dt(performance_ratio, &density_metrics);
        
        // Smoothly adapt toward optimal time step
        let dt_change = (optimal_dt - self.current_dt) * self.adaptation_rate;
        self.current_dt = (self.current_dt + dt_change).clamp(self.min_dt, self.max_dt);
    }
    
    /// Calculate particle density and interaction metrics
    fn calculate_density_metrics(&self, particles: &ParticleArrays) -> DensityMetrics {
        let total_particles = particles.count;
        
        if total_particles == 0 {
            return DensityMetrics::default();
        }
        
        // Count particles by LOD level
        let mut lod_counts = [0; 4];
        for &lod in &particles.lod_levels {
            if (lod as usize) < 4 {
                lod_counts[lod as usize] += 1;
            }
        }
        
        let active_particles = lod_counts[0] + lod_counts[1] + lod_counts[2]; // Exclude culled
        let high_lod_particles = lod_counts[0];
        
        // Calculate density ratios
        let active_ratio = active_particles as f64 / total_particles as f64;
        let high_lod_ratio = high_lod_particles as f64 / total_particles as f64;
        
        // Calculate average velocity (kinetic energy proxy)
        let total_velocity_sq: f64 = (0..particles.count)
            .map(|i| {
                let vx = particles.velocities_x[i];
                let vy = particles.velocities_y[i];
                let vz = particles.velocities_z[i];
                vx * vx + vy * vy + vz * vz
            })
            .sum();
        
        let avg_velocity_magnitude = if total_particles > 0 {
            (total_velocity_sq / total_particles as f64).sqrt()
        } else {
            0.0
        };
        
        // Calculate spatial clustering (simple version)
        let clustering_factor = self.calculate_clustering_factor(particles);
        
        DensityMetrics {
            total_particles,
            active_particles,
            high_lod_particles,
            active_ratio,
            high_lod_ratio,
            avg_velocity_magnitude,
            clustering_factor,
        }
    }
    
    /// Calculate spatial clustering factor (0.0 = uniform, 1.0 = highly clustered)
    fn calculate_clustering_factor(&self, particles: &ParticleArrays) -> f64 {
        if particles.count < 10 {
            return 0.0;
        }
        
        // Sample a subset of particles to avoid O(n²) computation
        let sample_size = (particles.count / 10).min(100).max(10);
        let step = particles.count / sample_size;
        
        let mut total_local_density = 0.0;
        let search_radius = 0.3; // Local neighborhood radius
        let search_radius_sq = search_radius * search_radius;
        
        for i in (0..particles.count).step_by(step).take(sample_size) {
            let mut local_count = 0;
            let pos_x = particles.positions_x[i];
            let pos_y = particles.positions_y[i];
            
            // Count particles within search radius
            for j in 0..particles.count {
                if i == j { continue; }
                
                let dx = particles.positions_x[j] - pos_x;
                let dy = particles.positions_y[j] - pos_y;
                let dist_sq = dx * dx + dy * dy;
                
                if dist_sq < search_radius_sq {
                    local_count += 1;
                }
            }
            
            total_local_density += local_count as f64;
        }
        
        let avg_local_density = total_local_density / sample_size as f64;
        let max_possible_density = particles.count as f64 * (search_radius_sq * std::f64::consts::PI / 16.0); // Approximate for 4x4 world
        
        (avg_local_density / max_possible_density).min(1.0)
    }
    
    /// Calculate optimal time step based on performance and density
    fn calculate_optimal_dt(&self, performance_ratio: f64, density: &DensityMetrics) -> f64 {
        let mut target_dt = self.base_dt;
        
        // Performance-based adjustment
        if performance_ratio > 1.0 + self.stability_buffer {
            // Running slow, increase time step to maintain frame rate
            target_dt *= 1.0 + (performance_ratio - 1.0) * 0.5;
        } else if performance_ratio < 1.0 - self.stability_buffer {
            // Running fast, can decrease time step for better accuracy
            target_dt *= performance_ratio;
        }
        
        // Density-based adjustment
        if density.active_ratio > self.high_density_threshold {
            // High particle density, may need smaller time steps for stability
            let density_factor = 1.0 - (density.active_ratio - self.high_density_threshold) * 0.5;
            target_dt *= density_factor;
        } else if density.active_ratio < self.low_density_threshold {
            // Low particle density, can use larger time steps
            let density_factor = 1.0 + (self.low_density_threshold - density.active_ratio) * 0.3;
            target_dt *= density_factor;
        }
        
        // Velocity-based adjustment (higher velocities need smaller time steps)
        if density.avg_velocity_magnitude > 1.0 {
            let velocity_factor = 1.0 / (1.0 + density.avg_velocity_magnitude * 0.1);
            target_dt *= velocity_factor;
        }
        
        // Clustering-based adjustment (clustered particles need smaller time steps)
        if density.clustering_factor > 0.5 {
            let clustering_factor = 1.0 - (density.clustering_factor - 0.5) * 0.3;
            target_dt *= clustering_factor;
        }
        
        target_dt.clamp(self.min_dt, self.max_dt)
    }
    
    /// Get current time step
    pub fn get_dt(&self) -> f64 {
        self.current_dt
    }
    
    /// Get performance metrics for display
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let target_frame_time = 1.0 / self.target_fps;
        let current_fps = if self.avg_frame_time > 0.0 {
            1.0 / self.avg_frame_time
        } else {
            self.target_fps
        };
        
        PerformanceMetrics {
            current_fps,
            target_fps: self.target_fps,
            avg_frame_time: self.avg_frame_time,
            target_frame_time,
            current_dt: self.current_dt,
            base_dt: self.base_dt,
            dt_efficiency: self.current_dt / self.base_dt,
        }
    }
    
    /// Force a specific time step (for manual control)
    pub fn set_dt(&mut self, dt: f64) {
        self.current_dt = dt.clamp(self.min_dt, self.max_dt);
    }
    
    /// Reset adaptation (useful when simulation parameters change significantly)
    pub fn reset(&mut self) {
        self.current_dt = self.base_dt;
        self.frame_times.clear();
        self.avg_frame_time = 1.0 / self.target_fps;
    }
}

#[derive(Debug, Clone)]
pub struct DensityMetrics {
    pub total_particles: usize,
    pub active_particles: usize,
    pub high_lod_particles: usize,
    pub active_ratio: f64,
    pub high_lod_ratio: f64,
    pub avg_velocity_magnitude: f64,
    pub clustering_factor: f64,
}

impl Default for DensityMetrics {
    fn default() -> Self {
        Self {
            total_particles: 0,
            active_particles: 0,
            high_lod_particles: 0,
            active_ratio: 0.0,
            high_lod_ratio: 0.0,
            avg_velocity_magnitude: 0.0,
            clustering_factor: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub current_fps: f64,
    pub target_fps: f64,
    pub avg_frame_time: f64,
    pub target_frame_time: f64,
    pub current_dt: f64,
    pub base_dt: f64,
    pub dt_efficiency: f64,
}

/// Simplified Barnes-Hut style hierarchical force approximation
pub struct HierarchicalForceApproximator {
    /// Quadtree for spatial partitioning
    quadtree: QuadTree,
    /// Theta parameter for approximation quality (0.0 = exact, 1.0 = very approximate)
    pub theta: f64,
}

impl HierarchicalForceApproximator {
    pub fn new(world_bounds: (f64, f64, f64, f64), theta: f64) -> Self {
        Self {
            quadtree: QuadTree::new(world_bounds),
            theta,
        }
    }
    
    /// Build the quadtree from particle positions
    pub fn build_tree(&mut self, particles: &ParticleArrays) {
        self.quadtree.clear();
        
        for i in 0..particles.count {
            // Only add high and medium LOD particles to tree
            if particles.lod_levels[i] <= 1 {
                self.quadtree.insert(i, particles.positions_x[i], particles.positions_y[i]);
            }
        }
        
        self.quadtree.calculate_centers_of_mass(particles);
    }
    
    /// Calculate force on a particle using hierarchical approximation
    pub fn calculate_force(&self, particle_idx: usize, particles: &ParticleArrays, 
                          interaction_matrix: &[Vec<f64>], rmax: f64, force_multiplier: f64) -> (f64, f64, f64) {
        let pos_x = particles.positions_x[particle_idx];
        let pos_y = particles.positions_y[particle_idx];
        let type_i = particles.type_ids[particle_idx] as usize;
        
        self.quadtree.calculate_force_recursive(
            pos_x, pos_y, type_i, particles, interaction_matrix, 
            rmax, force_multiplier, self.theta
        )
    }
}

/// Simplified quadtree for hierarchical force calculation
struct QuadTree {
    nodes: Vec<QuadNode>,
    root_bounds: (f64, f64, f64, f64), // (x_min, x_max, y_min, y_max)
}

struct QuadNode {
    bounds: (f64, f64, f64, f64),
    center_of_mass: (f64, f64),
    total_mass: f64,
    particle_indices: Vec<usize>,
    children: Option<[usize; 4]>, // Indices of child nodes
}

impl QuadTree {
    fn new(bounds: (f64, f64, f64, f64)) -> Self {
        Self {
            nodes: vec![QuadNode {
                bounds,
                center_of_mass: (0.0, 0.0),
                total_mass: 0.0,
                particle_indices: Vec::new(),
                children: None,
            }],
            root_bounds: bounds,
        }
    }
    
    fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(QuadNode {
            bounds: self.root_bounds,
            center_of_mass: (0.0, 0.0),
            total_mass: 0.0,
            particle_indices: Vec::new(),
            children: None,
        });
    }
    
    fn insert(&mut self, particle_idx: usize, x: f64, y: f64) {
        self.insert_recursive(0, particle_idx, x, y);
    }
    
    fn insert_recursive(&mut self, node_idx: usize, particle_idx: usize, x: f64, y: f64) {
        // Simplified insertion - would need full implementation for production use
        self.nodes[node_idx].particle_indices.push(particle_idx);
    }
    
    fn calculate_centers_of_mass(&mut self, particles: &ParticleArrays) {
        // Calculate center of mass for each node (simplified)
        for node in &mut self.nodes {
            if node.particle_indices.is_empty() {
                continue;
            }
            
            let mut total_x = 0.0;
            let mut total_y = 0.0;
            let mut count = 0.0;
            
            for &idx in &node.particle_indices {
                total_x += particles.positions_x[idx];
                total_y += particles.positions_y[idx];
                count += 1.0;
            }
            
            node.center_of_mass = (total_x / count, total_y / count);
            node.total_mass = count;
        }
    }
    
    fn calculate_force_recursive(&self, x: f64, y: f64, type_id: usize, 
                                particles: &ParticleArrays, interaction_matrix: &[Vec<f64>],
                                rmax: f64, force_multiplier: f64, theta: f64) -> (f64, f64, f64) {
        // Simplified force calculation - would need full Barnes-Hut implementation
        (0.0, 0.0, 0.0)
    }
}