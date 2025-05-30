// Structure of Arrays (SoA) optimized particle system
use bytemuck::{Pod, Zeroable};
use nalgebra::Vector3;
use rayon::prelude::*;

/// SoA particle data for better cache locality and SIMD optimization
#[derive(Debug, Clone)]
pub struct ParticleArrays {
    /// X positions of all particles
    pub positions_x: Vec<f64>,
    /// Y positions of all particles  
    pub positions_y: Vec<f64>,
    /// Z positions of all particles
    pub positions_z: Vec<f64>,
    
    /// X velocities of all particles
    pub velocities_x: Vec<f64>,
    /// Y velocities of all particles
    pub velocities_y: Vec<f64>,
    /// Z velocities of all particles
    pub velocities_z: Vec<f64>,
    
    /// Type IDs of all particles
    pub type_ids: Vec<u32>,
    
    /// LOD levels of all particles (0=high, 1=medium, 2=low, 3=culled)
    pub lod_levels: Vec<u32>,
    
    /// Force accumulation buffers
    pub forces_x: Vec<f64>,
    pub forces_y: Vec<f64>,
    pub forces_z: Vec<f64>,
    
    /// Particle count
    pub count: usize,
}

impl ParticleArrays {
    pub fn new(capacity: usize) -> Self {
        Self {
            positions_x: Vec::with_capacity(capacity),
            positions_y: Vec::with_capacity(capacity),
            positions_z: Vec::with_capacity(capacity),
            velocities_x: Vec::with_capacity(capacity),
            velocities_y: Vec::with_capacity(capacity),
            velocities_z: Vec::with_capacity(capacity),
            type_ids: Vec::with_capacity(capacity),
            lod_levels: Vec::with_capacity(capacity),
            forces_x: Vec::with_capacity(capacity),
            forces_y: Vec::with_capacity(capacity),
            forces_z: Vec::with_capacity(capacity),
            count: 0,
        }
    }
    
    pub fn resize(&mut self, new_count: usize) {
        self.positions_x.resize(new_count, 0.0);
        self.positions_y.resize(new_count, 0.0);
        self.positions_z.resize(new_count, 0.0);
        self.velocities_x.resize(new_count, 0.0);
        self.velocities_y.resize(new_count, 0.0);
        self.velocities_z.resize(new_count, 0.0);
        self.type_ids.resize(new_count, 0);
        self.lod_levels.resize(new_count, 0);
        self.forces_x.resize(new_count, 0.0);
        self.forces_y.resize(new_count, 0.0);
        self.forces_z.resize(new_count, 0.0);
        self.count = new_count;
    }
    
    pub fn clear_forces(&mut self) {
        // Vectorized force clearing
        self.forces_x.par_iter_mut().for_each(|f| *f = 0.0);
        self.forces_y.par_iter_mut().for_each(|f| *f = 0.0);
        self.forces_z.par_iter_mut().for_each(|f| *f = 0.0);
    }
    
    pub fn add_particle(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, type_id: u32) {
        self.positions_x.push(position.x);
        self.positions_y.push(position.y);
        self.positions_z.push(position.z);
        self.velocities_x.push(velocity.x);
        self.velocities_y.push(velocity.y);
        self.velocities_z.push(velocity.z);
        self.type_ids.push(type_id);
        self.lod_levels.push(0);
        self.forces_x.push(0.0);
        self.forces_y.push(0.0);
        self.forces_z.push(0.0);
        self.count += 1;
    }
    
    pub fn get_position(&self, index: usize) -> Vector3<f64> {
        Vector3::new(
            self.positions_x[index],
            self.positions_y[index],
            self.positions_z[index],
        )
    }
    
    pub fn get_velocity(&self, index: usize) -> Vector3<f64> {
        Vector3::new(
            self.velocities_x[index],
            self.velocities_y[index],
            self.velocities_z[index],
        )
    }
    
    pub fn set_position(&mut self, index: usize, position: Vector3<f64>) {
        self.positions_x[index] = position.x;
        self.positions_y[index] = position.y;
        self.positions_z[index] = position.z;
    }
    
    pub fn set_velocity(&mut self, index: usize, velocity: Vector3<f64>) {
        self.velocities_x[index] = velocity.x;
        self.velocities_y[index] = velocity.y;
        self.velocities_z[index] = velocity.z;
    }
    
    pub fn add_force(&mut self, index: usize, force: Vector3<f64>) {
        self.forces_x[index] += force.x;
        self.forces_y[index] += force.y;
        self.forces_z[index] += force.z;
    }
    
    /// Calculate LOD levels based on camera position and view frustum
    pub fn calculate_lod_levels(&mut self, camera_pos: Vector3<f64>, camera_size: f64, 
                               frustum_bounds: Option<(f64, f64, f64, f64)>, // left, right, bottom, top
                               lod_distances: (f64, f64, f64)) {
        let (lod_dist_0, lod_dist_1, lod_dist_2) = lod_distances;
        
        for i in 0..self.count {
            let pos_x = self.positions_x[i];
            let pos_y = self.positions_y[i];
            
            // Check frustum culling first
            if let Some((left, right, bottom, top)) = frustum_bounds {
                if pos_x < left || pos_x > right || pos_y < bottom || pos_y > top {
                    self.lod_levels[i] = 3; // Culled
                    continue;
                }
            }
            
            // Calculate distance from camera
            let dx = pos_x - camera_pos.x;
            let dy = pos_y - camera_pos.y;
            let distance = (dx * dx + dy * dy).sqrt();
            
            // Scale distance by camera zoom level
            let scaled_distance = distance / camera_size;
            
            self.lod_levels[i] = if scaled_distance < lod_dist_0 {
                0 // High LOD
            } else if scaled_distance < lod_dist_1 {
                1 // Medium LOD
            } else if scaled_distance < lod_dist_2 {
                2 // Low LOD
            } else {
                3 // Culled
            };
        }
    }
    
    /// SIMD-optimized force calculation for high-LOD particles
    pub fn calculate_forces_vectorized(&mut self, interaction_matrix: &[Vec<f64>], 
                                     rmax: f64, force_multiplier: f64,
                                     spatial_grid: &SpatialGridSoA) {
        let rmax_sq = rmax * rmax;
        let beta = 0.3;
        
        // Process particles sequentially to avoid borrowing issues
        for i in 0..self.count {
            // Skip culled particles but allow all other LOD levels
            if self.lod_levels[i] == 3 {
                continue;
            }
            
            let mut force_x = 0.0;
            let mut force_y = 0.0;
            let mut force_z = 0.0;
            
            let pos_i_x = self.positions_x[i];
            let pos_i_y = self.positions_y[i];
            let pos_i_z = self.positions_z[i];
            let type_i = self.type_ids[i] as usize;
            
            // Get nearby particles from spatial grid
            let neighbors = spatial_grid.get_neighbors(pos_i_x, pos_i_y, rmax);
            
            for &j in &neighbors {
                if i == j {
                    continue;
                }
                
                // Skip interactions with culled particles
                if self.lod_levels[j] == 3 {
                    continue;
                }
                
                let mut delta_x = self.positions_x[j] - pos_i_x;
                let mut delta_y = self.positions_y[j] - pos_i_y;
                let delta_z = self.positions_z[j] - pos_i_z;
                
                // Handle wrapping boundaries
                let world_size = 4.0;
                let half_world = world_size * 0.5;
                
                if delta_x > half_world { delta_x -= world_size; }
                else if delta_x < -half_world { delta_x += world_size; }
                
                if delta_y > half_world { delta_y -= world_size; }
                else if delta_y < -half_world { delta_y += world_size; }
                
                let distance_sq = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                
                if distance_sq > 0.0 && distance_sq < rmax_sq {
                    let distance = distance_sq.sqrt();
                    let type_j = self.type_ids[j] as usize;
                    let attraction = interaction_matrix[type_i][type_j];
                    
                    let force_magnitude = if distance < beta * rmax {
                        (distance / (beta * rmax) - 1.0) * force_multiplier
                    } else {
                        attraction * (1.0 - (1.0 + beta - 2.0 * distance / rmax).abs() / (1.0 - beta)) * force_multiplier
                    };
                    
                    let inv_distance = 1.0 / distance;
                    force_x += delta_x * force_magnitude * inv_distance;
                    force_y += delta_y * force_magnitude * inv_distance;
                    force_z += delta_z * force_magnitude * inv_distance;
                }
            }
            
            self.forces_x[i] += force_x;
            self.forces_y[i] += force_y;
            self.forces_z[i] += force_z;
        }
    }
    
    /// Update positions and velocities with LOD-aware time stepping
    pub fn update_particles_vectorized(&mut self, dt: f64, friction: f64, wrap_boundaries: bool) {
        let friction_60 = friction.powf(dt * 60.0);
        
        // Sequential update to avoid borrowing issues
        for i in 0..self.count {
            // Skip culled particles
            if self.lod_levels[i] == 3 {
                continue;
            }
            
            // LOD-based time scaling
            let effective_dt = match self.lod_levels[i] {
                0 => dt,        // High LOD - full update rate
                1 => dt * 2.0,  // Medium LOD - compensate for half update rate
                2 => dt * 4.0,  // Low LOD - compensate for quarter update rate
                _ => continue,
            };
            
            // Update velocity with force and friction
            self.velocities_x[i] += self.forces_x[i] * effective_dt;
            self.velocities_y[i] += self.forces_y[i] * effective_dt;
            self.velocities_z[i] += self.forces_z[i] * effective_dt;
            
            self.velocities_x[i] *= friction_60;
            self.velocities_y[i] *= friction_60;
            self.velocities_z[i] *= friction_60;
            
            // Update position
            self.positions_x[i] += self.velocities_x[i] * effective_dt;
            self.positions_y[i] += self.velocities_y[i] * effective_dt;
            self.positions_z[i] += self.velocities_z[i] * effective_dt;
            
            // Handle boundaries
            if wrap_boundaries {
                let world_size = 4.0;
                let world_min = -2.0;
                
                self.positions_x[i] = world_min + ((self.positions_x[i] - world_min).rem_euclid(world_size));
                self.positions_y[i] = world_min + ((self.positions_y[i] - world_min).rem_euclid(world_size));
            } else {
                self.positions_x[i] = self.positions_x[i].clamp(-2.0, 2.0);
                self.positions_y[i] = self.positions_y[i].clamp(-2.0, 2.0);
            }
            
            self.positions_z[i] = 0.0;
        }
    }
}

/// Optimized spatial grid using SoA layout
pub struct SpatialGridSoA {
    grid: Vec<Vec<usize>>,
    grid_size: usize,
    cell_size: f64,
    world_min: f64,
}

impl SpatialGridSoA {
    pub fn new(cell_size: f64, grid_size: usize, world_min: f64) -> Self {
        Self {
            grid: vec![Vec::new(); grid_size * grid_size],
            grid_size,
            cell_size,
            world_min,
        }
    }
    
    pub fn clear(&mut self) {
        for cell in &mut self.grid {
            cell.clear();
        }
    }
    
    fn get_cell_index(&self, x: f64, y: f64) -> usize {
        let shifted_x = x - self.world_min;
        let shifted_y = y - self.world_min;
        let gx = ((shifted_x / self.cell_size).floor() as usize).min(self.grid_size - 1);
        let gy = ((shifted_y / self.cell_size).floor() as usize).min(self.grid_size - 1);
        gy * self.grid_size + gx
    }
    
    pub fn rebuild(&mut self, particles: &ParticleArrays) {
        self.clear();
        
        for i in 0..particles.count {
            // Only add non-culled particles to grid
            if particles.lod_levels[i] != 3 {
                let cell_idx = self.get_cell_index(particles.positions_x[i], particles.positions_y[i]);
                self.grid[cell_idx].push(i);
            }
        }
    }
    
    pub fn get_neighbors(&self, x: f64, y: f64, range: f64) -> Vec<usize> {
        let mut neighbors = Vec::new();
        
        let min_x = (x - range).max(self.world_min);
        let max_x = (x + range).min(self.world_min + 4.0);
        let min_y = (y - range).max(self.world_min);
        let max_y = (y + range).min(self.world_min + 4.0);
        
        let shifted_min_x = min_x - self.world_min;
        let shifted_max_x = max_x - self.world_min;
        let shifted_min_y = min_y - self.world_min;
        let shifted_max_y = max_y - self.world_min;
        
        let min_gx = ((shifted_min_x / self.cell_size).floor() as usize).min(self.grid_size - 1);
        let max_gx = ((shifted_max_x / self.cell_size).floor() as usize).min(self.grid_size - 1);
        let min_gy = ((shifted_min_y / self.cell_size).floor() as usize).min(self.grid_size - 1);
        let max_gy = ((shifted_max_y / self.cell_size).floor() as usize).min(self.grid_size - 1);
        
        for gy in min_gy..=max_gy {
            for gx in min_gx..=max_gx {
                let cell_idx = gy * self.grid_size + gx;
                neighbors.extend(&self.grid[cell_idx]);
            }
        }
        
        neighbors
    }
}