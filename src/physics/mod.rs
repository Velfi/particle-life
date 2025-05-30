use nalgebra::Vector3;
use rand::Rng;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

pub type Position = Vector3<f64>;
pub type Velocity = Vector3<f64>;

#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Position,
    pub velocity: Velocity,
    pub type_id: usize,
}

impl Particle {
    pub fn new() -> Self {
        Self {
            position: Position::new(0.0, 0.0, 0.0),
            velocity: Velocity::new(0.0, 0.0, 0.0),
            type_id: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsSettings {
    pub wrap: bool,
    pub rmax: f64,
    pub friction: f64,
    pub force: f64,
    pub dt: f64,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            wrap: true,
            rmax: 0.03,
            friction: 0.85,
            force: 1.0,
            dt: 0.016,
        }
    }
}

pub trait PositionSetter: Send + Sync {
    fn set_position(&self, position: &mut Position, type_id: usize, n_types: usize);
}

pub trait TypeSetter: Send + Sync {
    fn set_type(&self, position: &Position, velocity: &Velocity, current_type: usize, n_types: usize) -> usize;
}

pub trait MatrixGenerator: Send + Sync {
    fn generate(&self, size: usize) -> Matrix;
}

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<Vec<f64>>,
    size: usize,
}

impl Matrix {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![vec![0.0; size]; size],
            size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i][j] = value;
    }

    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        self.data.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|val| {
                *val = rng.gen_range(-1.0..1.0);
            });
        });
    }
}

pub struct PhysicsSnapshot {
    pub positions: Vec<Position>,
    pub velocities: Vec<Velocity>,
    pub types: Vec<usize>,
    pub particle_count: usize,
    pub type_count: Vec<usize>,
}

impl PhysicsSnapshot {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            types: Vec::new(),
            particle_count: 0,
            type_count: Vec::new(),
        }
    }
}

pub struct ExtendedPhysics {
    pub particles: Vec<Particle>,
    pub settings: PhysicsSettings,
    pub matrix: Matrix,
    position_setter: Box<dyn PositionSetter>,
    matrix_generator: Box<dyn MatrixGenerator>,
    spatial_grid: SpatialGrid,
    force_buffer: Vec<Vector3<f64>>,
}

impl ExtendedPhysics {
    pub fn new(
        position_setter: Box<dyn PositionSetter>,
        matrix_generator: Box<dyn MatrixGenerator>,
    ) -> Self {
        let matrix = matrix_generator.generate(2);
        let grid_size = 32; // 32x32 grid for spatial partitioning
        let cell_size = 4.0 / grid_size as f64; // Adjust for larger world (-2.0 to 2.0 = 4.0 width)
        Self {
            particles: Vec::new(),
            settings: PhysicsSettings::default(),
            matrix,
            position_setter,
            matrix_generator,
            spatial_grid: SpatialGrid::new(cell_size, grid_size),
            force_buffer: Vec::new(),
        }
    }

    pub fn set_particle_count(&mut self, count: usize) {
        self.particles.clear();
        self.force_buffer.clear();
        self.force_buffer.resize(count, Vector3::new(0.0, 0.0, 0.0));
        
        for i in 0..count {
            let mut particle = Particle::new();
            let type_id = i % self.matrix.size();
            self.position_setter.set_position(&mut particle.position, type_id, self.matrix.size());
            particle.type_id = type_id;
            self.particles.push(particle);
        }
    }

    pub fn set_matrix_size(&mut self, size: usize) {
        self.matrix = self.matrix_generator.generate(size);
        self.ensure_types();
    }

    pub fn ensure_types(&mut self) {
        for particle in &mut self.particles {
            if particle.type_id >= self.matrix.size() {
                particle.type_id = 0;
            }
        }
    }

    pub fn set_positions(&mut self) {
        for particle in &mut self.particles {
            self.position_setter.set_position(&mut particle.position, particle.type_id, self.matrix.size());
        }
    }

    pub fn set_positions_with_setter(&mut self, position_setter: &dyn PositionSetter) {
        for particle in &mut self.particles {
            position_setter.set_position(&mut particle.position, particle.type_id, self.matrix.size());
        }
    }

    pub fn set_types_with_setter(&mut self, type_setter: &dyn TypeSetter) {
        for particle in &mut self.particles {
            particle.type_id = type_setter.set_type(
                &particle.position,
                &particle.velocity,
                particle.type_id,
                self.matrix.size(),
            );
        }
    }

    pub fn generate_matrix(&mut self) {
        self.matrix = self.matrix_generator.generate(self.matrix.size());
    }

    pub fn generate_matrix_with_generator(&mut self, matrix_generator: &dyn MatrixGenerator) {
        self.matrix = matrix_generator.generate(self.matrix.size());
    }

    pub fn get_type_count(&self) -> Vec<usize> {
        let mut type_count = vec![0; self.matrix.size()];
        for &type_id in self.particles.iter().map(|p| &p.type_id) {
            if type_id < type_count.len() {
                type_count[type_id] += 1;
            }
        }
        type_count
    }

    pub fn set_type_count(&mut self, new_type_count: &[usize]) {
        self.particles.clear();
        
        for (type_id, &count) in new_type_count.iter().enumerate() {
            for _ in 0..count {
                let mut particle = Particle::new();
                self.position_setter.set_position(&mut particle.position, type_id, self.matrix.size());
                particle.type_id = type_id;
                self.particles.push(particle);
            }
        }
    }

    pub fn set_type_count_equal(&mut self) {
        let total_particles = self.particles.len();
        let types_count = self.matrix.size();
        let particles_per_type = total_particles / types_count;
        let remainder = total_particles % types_count;
        
        let new_type_count: Vec<usize> = (0..types_count)
            .map(|i| particles_per_type + if i < remainder { 1 } else { 0 })
            .collect();
        
        self.set_type_count(&new_type_count);
    }

    pub fn update(&mut self) {
        self.update_with_cursor(None, 0.0, 0.0);
    }
    
    pub fn update_with_cursor(&mut self, cursor_pos: Option<Vector3<f64>>, cursor_size: f64, cursor_strength: f64) {
        let dt = self.settings.dt;
        let rmax = self.settings.rmax;
        let friction = self.settings.friction;
        let force_multiplier = self.settings.force;
        let wrap = self.settings.wrap;

        // Early exit if no particles
        if self.particles.is_empty() {
            return;
        }

        // Ensure force buffer is the right size
        if self.force_buffer.len() != self.particles.len() {
            self.force_buffer.resize(self.particles.len(), Vector3::new(0.0, 0.0, 0.0));
        }

        // Clear spatial grid and rebuild it
        self.spatial_grid.clear();
        for (i, particle) in self.particles.iter().enumerate() {
            self.spatial_grid.insert(i, &particle.position);
        }

        // Clear force buffer
        for force in &mut self.force_buffer {
            *force = Vector3::new(0.0, 0.0, 0.0);
        }

        // Use parallel chunks for force calculation to work around borrow checker
        let chunk_size = (self.particles.len() / num_cpus::get()).max(1);
        let rmax_sq = rmax * rmax;
        let beta = 0.3;
        
        self.force_buffer.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, force_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let mut neighbor_buffer = Vec::new();
            
            for (local_i, force) in force_chunk.iter_mut().enumerate() {
                let i = start_idx + local_i;
                if i >= self.particles.len() { break; }
                
                let particle_i = &self.particles[i];
                
                // Get nearby particles from spatial grid
                self.spatial_grid.get_neighbors(&particle_i.position, rmax, &mut neighbor_buffer);
                
                for &j in &neighbor_buffer {
                    if i != j {
                        let particle_j = &self.particles[j];
                        let mut delta = particle_j.position - particle_i.position;
                        
                        // Handle wrapping boundaries with proper distance calculation
                        if wrap {
                            // Find the shortest distance across boundaries
                            let world_size = 4.0; // -2.0 to 2.0 = 4.0 width
                            
                            if delta.x > world_size / 2.0 {
                                delta.x -= world_size;
                            } else if delta.x < -world_size / 2.0 {
                                delta.x += world_size;
                            }
                            
                            if delta.y > world_size / 2.0 {
                                delta.y -= world_size;
                            } else if delta.y < -world_size / 2.0 {
                                delta.y += world_size;
                            }
                        }
                        
                        let distance_sq = delta.norm_squared();
                        
                        if distance_sq > 0.0 && distance_sq < rmax_sq {
                            let distance = distance_sq.sqrt();
                            let attraction = self.matrix.get(particle_i.type_id, particle_j.type_id);
                            
                            let force_magnitude = if distance < beta * rmax {
                                (distance / (beta * rmax) - 1.0) * force_multiplier
                            } else {
                                attraction * (1.0 - (1.0 + beta - 2.0 * distance / rmax).abs() / (1.0 - beta)) * force_multiplier
                            };
                            
                            *force += delta * (force_magnitude / distance);
                        }
                    }
                }
            }
        });

        // Apply cursor forces if cursor is active
        if let Some(cursor_position) = cursor_pos {
            let cursor_radius_sq = cursor_size * cursor_size;
            
            self.force_buffer.par_iter_mut().enumerate().for_each(|(i, force)| {
                let particle = &self.particles[i];
                let mut delta = cursor_position - particle.position;
                
                // Handle wrapping for cursor forces too
                if wrap {
                    let world_size = 4.0;
                    if delta.x > world_size / 2.0 {
                        delta.x -= world_size;
                    } else if delta.x < -world_size / 2.0 {
                        delta.x += world_size;
                    }
                    if delta.y > world_size / 2.0 {
                        delta.y -= world_size;
                    } else if delta.y < -world_size / 2.0 {
                        delta.y += world_size;
                    }
                }
                
                let distance_sq = delta.norm_squared();
                
                if distance_sq > 0.0 && distance_sq < cursor_radius_sq {
                    let distance = distance_sq.sqrt();
                    let falloff = 1.0 - (distance / cursor_size); // Linear falloff
                    let cursor_force_magnitude = cursor_strength * falloff;
                    
                    // Normalize delta to get direction
                    if distance > 0.0 {
                        *force += delta * (cursor_force_magnitude / distance);
                    }
                }
            });
        }

        // Apply forces and update velocities and positions in parallel
        self.particles.par_iter_mut().enumerate().for_each(|(i, particle)| {
            // Update velocity with force and friction
            particle.velocity += self.force_buffer[i] * dt;
            particle.velocity *= friction.powf(dt * 60.0); // Friction adjusted for frame rate
            
            // Update position
            particle.position += particle.velocity * dt;
            
            // Handle boundaries with improved wrapping
            if wrap {
                let world_size = 4.0;
                let world_min = -2.0;
                
                // Proper modulo wrapping that handles negative numbers correctly
                particle.position.x = world_min + ((particle.position.x - world_min).rem_euclid(world_size));
                particle.position.y = world_min + ((particle.position.y - world_min).rem_euclid(world_size));
            } else {
                particle.position.x = particle.position.x.clamp(-2.0, 2.0);
                particle.position.y = particle.position.y.clamp(-2.0, 2.0);
            }
            particle.position.z = 0.0;
        });
    }

    pub fn take_snapshot(&self) -> PhysicsSnapshot {
        PhysicsSnapshot {
            positions: self.particles.iter().map(|p| p.position).collect(),
            velocities: self.particles.iter().map(|p| p.velocity).collect(),
            types: self.particles.iter().map(|p| p.type_id).collect(),
            particle_count: self.particles.len(),
            type_count: self.get_type_count(),
        }
    }
}

// Default implementations
pub struct DefaultPositionSetter;

impl PositionSetter for DefaultPositionSetter {
    fn set_position(&self, position: &mut Position, _type_id: usize, _n_types: usize) {
        let mut rng = rand::thread_rng();
        position.x = rng.gen_range(-2.0..2.0);
        position.y = rng.gen_range(-2.0..2.0);
        position.z = 0.0;
    }
}

pub struct RandomPositionSetter;

impl PositionSetter for RandomPositionSetter {
    fn set_position(&self, position: &mut Position, _type_id: usize, _n_types: usize) {
        let mut rng = rand::thread_rng();
        position.x = rng.gen_range(-2.0..2.0);
        position.y = rng.gen_range(-2.0..2.0);
        position.z = 0.0;
    }
}

pub struct CenterPositionSetter;

impl PositionSetter for CenterPositionSetter {
    fn set_position(&self, position: &mut Position, _type_id: usize, _n_types: usize) {
        let mut rng = rand::thread_rng();
        let scale = 1.2; // Increased scale for larger world
        position.x = rng.gen_range(-1.0..1.0) * scale;
        position.y = rng.gen_range(-1.0..1.0) * scale;
        position.z = 0.0;
    }
}

pub struct UniformCirclePositionSetter;

impl PositionSetter for UniformCirclePositionSetter {
    fn set_position(&self, position: &mut Position, _type_id: usize, _n_types: usize) {
        let mut rng = rand::thread_rng();
        let max_radius = 2.0; // Increased for larger world
        let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let radius = max_radius * rng.gen_range(0.0..1.0_f64).sqrt();
        position.x = angle.cos() * radius;
        position.y = angle.sin() * radius;
        position.z = 0.0;
    }
}

pub struct CenteredCirclePositionSetter;

impl PositionSetter for CenteredCirclePositionSetter {
    fn set_position(&self, position: &mut Position, _type_id: usize, _n_types: usize) {
        let mut rng = rand::thread_rng();
        let max_radius = 2.0;
        let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let radius = max_radius * rng.gen_range(0.0..1.0);
        position.x = angle.cos() * radius;
        position.y = angle.sin() * radius;
        position.z = 0.0;
    }
}

pub struct RingPositionSetter;

impl PositionSetter for RingPositionSetter {
    fn set_position(&self, position: &mut Position, _type_id: usize, _n_types: usize) {
        let mut rng = rand::thread_rng();
        let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let radius = 0.7 + 0.02 * rng.gen_range(-1.0..1.0);
        position.x = angle.cos() * radius;
        position.y = angle.sin() * radius;
        position.z = 0.0;
    }
}

pub struct RainbowRingPositionSetter;

impl PositionSetter for RainbowRingPositionSetter {
    fn set_position(&self, position: &mut Position, type_id: usize, n_types: usize) {
        let mut rng = rand::thread_rng();
        let angle = (0.3 * rng.gen_range(-1.0..1.0) + type_id as f64) / n_types as f64 * 2.0 * std::f64::consts::PI;
        let radius = 0.7 + 0.02 * rng.gen_range(-1.0..1.0);
        position.x = angle.cos() * radius;
        position.y = angle.sin() * radius;
        position.z = 0.0;
    }
}

pub struct ColorBattlePositionSetter;

impl PositionSetter for ColorBattlePositionSetter {
    fn set_position(&self, position: &mut Position, type_id: usize, n_types: usize) {
        let mut rng = rand::thread_rng();
        let center_angle = type_id as f64 / n_types as f64 * 2.0 * std::f64::consts::PI;
        let center_radius = 0.5;
        
        let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let radius = rng.gen_range(0.0..0.1);
        
        position.x = center_radius * center_angle.cos() + angle.cos() * radius;
        position.y = center_radius * center_angle.sin() + angle.sin() * radius;
        position.z = 0.0;
    }
}

pub struct ColorWheelPositionSetter;

impl PositionSetter for ColorWheelPositionSetter {
    fn set_position(&self, position: &mut Position, type_id: usize, n_types: usize) {
        let mut rng = rand::thread_rng();
        let center_angle = type_id as f64 / n_types as f64 * 2.0 * std::f64::consts::PI;
        let center_radius = 0.3;
        let individual_radius = 0.2;
        
        position.x = center_radius * center_angle.cos() + rng.gen_range(-1.0..1.0) * individual_radius;
        position.y = center_radius * center_angle.sin() + rng.gen_range(-1.0..1.0) * individual_radius;
        position.z = 0.0;
    }
}

pub struct LinePositionSetter;

impl PositionSetter for LinePositionSetter {
    fn set_position(&self, position: &mut Position, _type_id: usize, _n_types: usize) {
        let mut rng = rand::thread_rng();
        position.x = (2.0 * rng.gen_range(0.0..1.0) - 1.0) * 2.0;
        position.y = (2.0 * rng.gen_range(0.0..1.0) - 1.0) * 0.6;
        position.z = 0.0;
    }
}

pub struct SpiralPositionSetter;

impl PositionSetter for SpiralPositionSetter {
    fn set_position(&self, position: &mut Position, _type_id: usize, _n_types: usize) {
        let mut rng = rand::thread_rng();
        let max_rotations = 2.0;
        let f = rng.gen_range(0.0..1.0);
        let angle = max_rotations * 2.0 * std::f64::consts::PI * f;
        
        let spread = 0.5 * f.min(0.2);
        let radius = 0.9 * f + spread * rng.gen_range(-1.0..1.0) * spread;
        
        position.x = radius * angle.cos();
        position.y = radius * angle.sin();
        position.z = 0.0;
    }
}

pub struct RainbowSpiralPositionSetter;

impl PositionSetter for RainbowSpiralPositionSetter {
    fn set_position(&self, position: &mut Position, type_id: usize, n_types: usize) {
        let mut rng = rand::thread_rng();
        let max_rotations = 2.0;
        let type_spread = 0.3 / n_types as f64;
        let mut f = (type_id + 1) as f64 / (n_types + 2) as f64 + type_spread * rng.gen_range(-1.0..1.0);
        f = f.clamp(0.0, 1.0);
        
        let angle = max_rotations * 2.0 * std::f64::consts::PI * f;
        
        let spread = 0.5 * f.min(0.2);
        let radius = 0.9 * f + spread * rng.gen_range(-1.0..1.0) * spread;
        
        position.x = radius * angle.cos();
        position.y = radius * angle.sin();
        position.z = 0.0;
    }
}

pub struct DefaultMatrixGenerator;

impl MatrixGenerator for DefaultMatrixGenerator {
    fn generate(&self, size: usize) -> Matrix {
        let mut matrix = Matrix::new(size);
        matrix.randomize();
        matrix
    }
}

pub struct SymmetryMatrixGenerator;

impl MatrixGenerator for SymmetryMatrixGenerator {
    fn generate(&self, size: usize) -> Matrix {
        let mut matrix = Matrix::new(size);
        matrix.randomize();
        // Make symmetric
        for i in 0..size {
            for j in i..size {
                let value = matrix.get(j, i);
                matrix.set(i, j, value);
            }
        }
        matrix
    }
}

pub struct ChainsMatrixGenerator;

impl MatrixGenerator for ChainsMatrixGenerator {
    fn generate(&self, size: usize) -> Matrix {
        let mut matrix = Matrix::new(size);
        for i in 0..size {
            for j in 0..size {
                if j == i || j == (i + 1) % size || j == (i + size - 1) % size {
                    matrix.set(i, j, 1.0);
                } else {
                    matrix.set(i, j, -1.0);
                }
            }
        }
        matrix
    }
}

pub struct Chains2MatrixGenerator;

impl MatrixGenerator for Chains2MatrixGenerator {
    fn generate(&self, size: usize) -> Matrix {
        let mut matrix = Matrix::new(size);
        for i in 0..size {
            for j in 0..size {
                if j == i {
                    matrix.set(i, j, 1.0);
                } else if j == (i + 1) % size || j == (i + size - 1) % size {
                    matrix.set(i, j, 0.2);
                } else {
                    matrix.set(i, j, -1.0);
                }
            }
        }
        matrix
    }
}

pub struct Chains3MatrixGenerator;

impl MatrixGenerator for Chains3MatrixGenerator {
    fn generate(&self, size: usize) -> Matrix {
        let mut matrix = Matrix::new(size);
        for i in 0..size {
            for j in 0..size {
                if j == i {
                    matrix.set(i, j, 1.0);
                } else if j == (i + 1) % size || j == (i + size - 1) % size {
                    matrix.set(i, j, 0.2);
                } else {
                    matrix.set(i, j, 0.0);
                }
            }
        }
        matrix
    }
}

pub struct SnakesMatrixGenerator;

impl MatrixGenerator for SnakesMatrixGenerator {
    fn generate(&self, size: usize) -> Matrix {
        let mut matrix = Matrix::new(size);
        for i in 0..size {
            matrix.set(i, i, 1.0);
            matrix.set(i, (i + 1) % size, 0.2);
        }
        matrix
    }
}

pub struct ZeroMatrixGenerator;

impl MatrixGenerator for ZeroMatrixGenerator {
    fn generate(&self, size: usize) -> Matrix {
        Matrix::new(size) // Already initialized to zeros
    }
}

pub struct DefaultTypeSetter;

impl TypeSetter for DefaultTypeSetter {
    fn set_type(&self, _position: &Position, _velocity: &Velocity, _current_type: usize, n_types: usize) -> usize {
        let mut rng = rand::thread_rng();
        rng.gen_range(0..n_types)
    }
}

pub struct RandomTypeSetter;

impl TypeSetter for RandomTypeSetter {
    fn set_type(&self, _position: &Position, _velocity: &Velocity, _current_type: usize, n_types: usize) -> usize {
        let mut rng = rand::thread_rng();
        rng.gen_range(0..n_types)
    }
}

pub struct Randomize10PercentTypeSetter;

impl TypeSetter for Randomize10PercentTypeSetter {
    fn set_type(&self, _position: &Position, _velocity: &Velocity, current_type: usize, n_types: usize) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen_range(0.0..1.0) < 0.1 {
            Self::map_type(rng.gen_range(0.0..1.0), n_types)
        } else {
            current_type
        }
    }
}

impl Randomize10PercentTypeSetter {
    fn map_type(value: f64, n_types: usize) -> usize {
        ((value * n_types as f64).floor() as usize).min(n_types - 1)
    }
}

pub struct SlicesTypeSetter;

impl TypeSetter for SlicesTypeSetter {
    fn set_type(&self, position: &Position, _velocity: &Velocity, _current_type: usize, n_types: usize) -> usize {
        Self::map_type(position.x, n_types)
    }
}

impl SlicesTypeSetter {
    fn map_type(value: f64, n_types: usize) -> usize {
        ((value * n_types as f64).floor() as usize).min(n_types - 1)
    }
}

pub struct OnionTypeSetter;

impl TypeSetter for OnionTypeSetter {
    fn set_type(&self, position: &Position, _velocity: &Velocity, _current_type: usize, n_types: usize) -> usize {
        let center = Vector3::new(0.0, 0.0, 0.0);
        let distance = (position - center).norm() * 2.0;
        Self::map_type(distance, n_types)
    }
}

impl OnionTypeSetter {
    fn map_type(value: f64, n_types: usize) -> usize {
        ((value * n_types as f64).floor() as usize).min(n_types - 1)
    }
}

pub struct RotateTypeSetter;

impl TypeSetter for RotateTypeSetter {
    fn set_type(&self, _position: &Position, _velocity: &Velocity, current_type: usize, n_types: usize) -> usize {
        (current_type + 1) % n_types
    }
}

pub struct FlipTypeSetter;

impl TypeSetter for FlipTypeSetter {
    fn set_type(&self, _position: &Position, _velocity: &Velocity, current_type: usize, n_types: usize) -> usize {
        n_types - 1 - current_type
    }
}

pub struct MoreOfFirstTypeSetter;

impl TypeSetter for MoreOfFirstTypeSetter {
    fn set_type(&self, _position: &Position, _velocity: &Velocity, _current_type: usize, n_types: usize) -> usize {
        let mut rng = rand::thread_rng();
        let value = rng.gen_range(0.0..1.0) * rng.gen_range(0.0..1.0);
        Self::map_type(value, n_types)
    }
}

impl MoreOfFirstTypeSetter {
    fn map_type(value: f64, n_types: usize) -> usize {
        ((value * n_types as f64).floor() as usize).min(n_types - 1)
    }
}

pub struct KillStillTypeSetter;

impl TypeSetter for KillStillTypeSetter {
    fn set_type(&self, _position: &Position, velocity: &Velocity, current_type: usize, n_types: usize) -> usize {
        if velocity.norm() < 0.01 {
            n_types - 1
        } else {
            current_type
        }
    }
}

// Add num_cpus to Cargo.toml
extern crate num_cpus;

// Spatial grid for efficient neighbor finding
struct SpatialGrid {
    grid: Vec<Vec<usize>>,
    grid_size: usize,
    cell_size: f64,
}

impl SpatialGrid {
    fn new(cell_size: f64, grid_size: usize) -> Self {
        Self {
            grid: vec![Vec::new(); grid_size * grid_size],
            grid_size,
            cell_size,
        }
    }

    fn clear(&mut self) {
        for cell in &mut self.grid {
            cell.clear();
        }
    }

    fn get_cell_index(&self, x: f64, y: f64) -> usize {
        // Shift coordinates from [-2.0, 2.0] to [0.0, 4.0] for grid indexing
        let shifted_x = x + 2.0;
        let shifted_y = y + 2.0;
        let gx = ((shifted_x / self.cell_size).floor() as usize).min(self.grid_size - 1);
        let gy = ((shifted_y / self.cell_size).floor() as usize).min(self.grid_size - 1);
        gy * self.grid_size + gx
    }

    fn insert(&mut self, particle_idx: usize, position: &Position) {
        let cell_idx = self.get_cell_index(position.x, position.y);
        self.grid[cell_idx].push(particle_idx);
        
        // For wrapping boundaries, also insert into wrapped neighbor cells if near edges
        // This ensures particles near boundaries can interact with particles on the opposite side
        let world_size = 4.0;
        let edge_threshold = 0.1; // Distance from edge to consider for wrapping
        
        // Check if near left/right edges
        if position.x < -2.0 + edge_threshold {
            // Near left edge, also insert to right side
            let wrapped_x = position.x + world_size;
            let wrapped_cell_idx = self.get_cell_index(wrapped_x, position.y);
            if wrapped_cell_idx < self.grid.len() {
                self.grid[wrapped_cell_idx].push(particle_idx);
            }
        } else if position.x > 2.0 - edge_threshold {
            // Near right edge, also insert to left side
            let wrapped_x = position.x - world_size;
            let wrapped_cell_idx = self.get_cell_index(wrapped_x, position.y);
            if wrapped_cell_idx < self.grid.len() {
                self.grid[wrapped_cell_idx].push(particle_idx);
            }
        }
        
        // Check if near top/bottom edges
        if position.y < -2.0 + edge_threshold {
            // Near bottom edge, also insert to top side
            let wrapped_y = position.y + world_size;
            let wrapped_cell_idx = self.get_cell_index(position.x, wrapped_y);
            if wrapped_cell_idx < self.grid.len() {
                self.grid[wrapped_cell_idx].push(particle_idx);
            }
        } else if position.y > 2.0 - edge_threshold {
            // Near top edge, also insert to bottom side
            let wrapped_y = position.y - world_size;
            let wrapped_cell_idx = self.get_cell_index(position.x, wrapped_y);
            if wrapped_cell_idx < self.grid.len() {
                self.grid[wrapped_cell_idx].push(particle_idx);
            }
        }
    }

    fn get_neighbors(&self, position: &Position, range: f64, neighbors: &mut Vec<usize>) {
        neighbors.clear();
        
        // Calculate the grid cells that need to be searched
        let min_x = position.x - range;
        let max_x = position.x + range;
        let min_y = position.y - range;
        let max_y = position.y + range;
        
        // Convert to grid coordinates and handle wrapping
        let shifted_min_x = min_x + 2.0;
        let shifted_max_x = max_x + 2.0;
        let shifted_min_y = min_y + 2.0;
        let shifted_max_y = max_y + 2.0;
        
        let min_gx = (shifted_min_x / self.cell_size).floor() as i32;
        let max_gx = (shifted_max_x / self.cell_size).floor() as i32;
        let min_gy = (shifted_min_y / self.cell_size).floor() as i32;
        let max_gy = (shifted_max_y / self.cell_size).floor() as i32;
        
        // Search all relevant cells, including wrapped ones
        for gy in min_gy..=max_gy {
            for gx in min_gx..=max_gx {
                // Handle wrapping for cells that extend beyond boundaries
                let wrapped_gx = if gx < 0 {
                    (gx + self.grid_size as i32) % self.grid_size as i32
                } else if gx >= self.grid_size as i32 {
                    gx % self.grid_size as i32
                } else {
                    gx
                } as usize;
                
                let wrapped_gy = if gy < 0 {
                    (gy + self.grid_size as i32) % self.grid_size as i32
                } else if gy >= self.grid_size as i32 {
                    gy % self.grid_size as i32
                } else {
                    gy
                } as usize;
                
                if wrapped_gx < self.grid_size && wrapped_gy < self.grid_size {
                    let cell_idx = wrapped_gy * self.grid_size + wrapped_gx;
                    neighbors.extend(&self.grid[cell_idx]);
                }
            }
        }
    }
}