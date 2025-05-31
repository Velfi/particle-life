// Physics compute shader for particle life simulation
//
// This shader implements the core physics simulation on the GPU:
// - Spatial partitioning for efficient neighbor finding
// - Force calculations between particles
// - Position and velocity updates
// - World boundary handling
// - Cursor interaction forces
//
// The simulation uses a grid-based spatial partitioning system to efficiently
// find nearby particles and calculate forces between them.

/// Particle data structure
struct Particle {
    /// World position (x, y, z)
    position: vec3<f32>,
    /// Velocity vector (vx, vy, vz)
    velocity: vec3<f32>,
    /// Type ID for interaction rules
    type_id: u32,
    /// Padding for alignment
    _padding: u32,
}

/// Physics simulation settings
struct PhysicsSettings {
    /// Time step for simulation
    dt: f32,
    /// Maximum interaction radius
    rmax: f32,
    /// Velocity damping factor
    friction: f32,
    /// Force strength multiplier
    force_multiplier: f32,
    /// Size of the world
    world_size: f32,
    /// Whether to wrap particles around world (1) or not (0)
    wrap_boundaries: u32,
    /// Size of the interaction matrix
    matrix_size: u32,
    /// Whether cursor interaction is active (1) or not (0)
    cursor_active: u32,
    /// Cursor position in world space
    cursor_position: vec3<f32>,
    /// Radius of cursor influence
    cursor_size: f32,
    /// Strength of cursor force
    cursor_strength: f32,
    /// Padding for alignment
    _padding: vec3<f32>,
}

/// Parameters for spatial grid
struct SpatialGridParams {
    /// Size of each grid cell
    cell_size: f32,
    /// Number of cells in each dimension
    grid_size: u32,
    /// Minimum world coordinate
    world_min: f32,
    /// Padding for alignment
    _padding: f32,
}

// Storage buffers and uniforms
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec3<f32>>;
@group(0) @binding(2) var<uniform> settings: PhysicsSettings;
@group(0) @binding(3) var<storage, read> interaction_matrix: array<f32>;
@group(0) @binding(4) var<storage, read_write> spatial_grid: array<u32>;
@group(0) @binding(5) var<storage, read_write> grid_counts: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> grid_params: SpatialGridParams;

/// Size of each compute workgroup
const WORKGROUP_SIZE: u32 = 64u;
/// Maximum number of particles per grid cell
const MAX_PARTICLES_PER_CELL: u32 = 64u;

/// Calculates the grid cell index for a given position
/// 
/// Parameters:
/// - pos: World position to find cell for
/// 
/// Returns the index of the cell containing the position
fn get_cell_index(pos: vec3<f32>) -> u32 {
    let shifted_x = pos.x - grid_params.world_min;
    let shifted_y = pos.y - grid_params.world_min;
    
    let gx = u32(clamp(shifted_x / grid_params.cell_size, 0.0, f32(grid_params.grid_size - 1u)));
    let gy = u32(clamp(shifted_y / grid_params.cell_size, 0.0, f32(grid_params.grid_size - 1u)));
    
    return gy * grid_params.grid_size + gx;
}

/// Calculates the shortest distance between two positions, accounting for wrapping
/// 
/// Parameters:
/// - delta: Vector between two positions
/// 
/// Returns the wrapped distance vector
fn wrap_distance(delta: vec3<f32>) -> vec3<f32> {
    var result = delta;
    let half_world = settings.world_size * 0.5;
    
    if (settings.wrap_boundaries != 0u) {
        if (result.x > half_world) {
            result.x -= settings.world_size;
        } else if (result.x < -half_world) {
            result.x += settings.world_size;
        }
        
        if (result.y > half_world) {
            result.y -= settings.world_size;
        } else if (result.y < -half_world) {
            result.y += settings.world_size;
        }
    }
    
    return result;
}

/// Clears the spatial grid for the next frame
/// 
/// Resets particle counts and invalidates all grid entries
@compute @workgroup_size(WORKGROUP_SIZE)
fn clear_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_cells = grid_params.grid_size * grid_params.grid_size;
    
    if (index >= total_cells) {
        return;
    }
    
    atomicStore(&grid_counts[index], 0u);
    
    // Clear grid entries for this cell
    let base_idx = index * MAX_PARTICLES_PER_CELL;
    for (var i = 0u; i < MAX_PARTICLES_PER_CELL; i++) {
        spatial_grid[base_idx + i] = 0xFFFFFFFFu; // Invalid particle index
    }
}

/// Builds the spatial grid by assigning particles to cells
/// 
/// Each particle is assigned to the cell containing its position
@compute @workgroup_size(WORKGROUP_SIZE)
fn build_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    let particle_count = arrayLength(&particles);
    
    if (particle_idx >= particle_count) {
        return;
    }
    
    let particle = particles[particle_idx];
    let cell_idx = get_cell_index(particle.position);
    
    // Atomically increment grid count and get slot
    let slot = atomicAdd(&grid_counts[cell_idx], 1u);
    
    if (slot < MAX_PARTICLES_PER_CELL) {
        let grid_entry_idx = cell_idx * MAX_PARTICLES_PER_CELL + slot;
        spatial_grid[grid_entry_idx] = particle_idx;
    }
}

/// Calculates forces between particles using the spatial grid
/// 
/// For each particle:
/// 1. Find nearby particles using the grid
/// 2. Calculate forces based on distance and interaction matrix
/// 3. Add cursor forces if active
@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    let particle_count = arrayLength(&particles);
    
    if (particle_idx >= particle_count) {
        return;
    }
    
    let particle_i = particles[particle_idx];
    var total_force = vec3<f32>(0.0);
    
    let rmax_sq = settings.rmax * settings.rmax;
    let beta = 0.3; // Force curve parameter
    
    // Get nearby particles from spatial grid
    let particle_cell = get_cell_index(particle_i.position);
    let grid_size = grid_params.grid_size;
    
    // Check neighboring cells (3x3 grid around particle)
    for (var dy = -1i; dy <= 1i; dy++) {
        for (var dx = -1i; dx <= 1i; dx++) {
            let gx = i32(particle_cell % grid_size) + dx;
            let gy = i32(particle_cell / grid_size) + dy;
            
            // Handle wrapping
            var wrapped_gx = gx;
            var wrapped_gy = gy;
            
            if (settings.wrap_boundaries != 0u) {
                wrapped_gx = (gx + i32(grid_size)) % i32(grid_size);
                wrapped_gy = (gy + i32(grid_size)) % i32(grid_size);
            } else {
                if (gx < 0 || gx >= i32(grid_size) || gy < 0 || gy >= i32(grid_size)) {
                    continue;
                }
            }
            
            let neighbor_cell = u32(wrapped_gy) * grid_size + u32(wrapped_gx);
            let particles_in_cell = min(atomicLoad(&grid_counts[neighbor_cell]), MAX_PARTICLES_PER_CELL);
            
            // Check all particles in this cell
            for (var i = 0u; i < particles_in_cell; i++) {
                let neighbor_idx = spatial_grid[neighbor_cell * MAX_PARTICLES_PER_CELL + i];
                
                if (neighbor_idx == 0xFFFFFFFFu || neighbor_idx == particle_idx) {
                    continue;
                }
                
                let particle_j = particles[neighbor_idx];
                let delta = wrap_distance(particle_j.position - particle_i.position);
                let distance_sq = dot(delta, delta);
                
                if (distance_sq > 0.0 && distance_sq < rmax_sq) {
                    let distance = sqrt(distance_sq);
                    let matrix_idx = particle_i.type_id * settings.matrix_size + particle_j.type_id;
                    let attraction = interaction_matrix[matrix_idx];
                    
                    var force_magnitude: f32;
                    if (distance < beta * settings.rmax) {
                        // Repulsive force at close range
                        force_magnitude = (distance / (beta * settings.rmax) - 1.0) * settings.force_multiplier;
                    } else {
                        // Attractive/repulsive force based on interaction matrix
                        force_magnitude = attraction * (1.0 - abs(1.0 + beta - 2.0 * distance / settings.rmax) / (1.0 - beta)) * settings.force_multiplier;
                    }
                    
                    total_force += delta * (force_magnitude / distance);
                }
            }
        }
    }
    
    // Add cursor forces if active
    if (settings.cursor_active != 0u) {
        let cursor_delta = wrap_distance(settings.cursor_position - particle_i.position);
        let cursor_distance_sq = dot(cursor_delta, cursor_delta);
        let cursor_radius_sq = settings.cursor_size * settings.cursor_size;
        
        if (cursor_distance_sq > 0.0 && cursor_distance_sq < cursor_radius_sq) {
            let cursor_distance = sqrt(cursor_distance_sq);
            let falloff = 1.0 - (cursor_distance / settings.cursor_size);
            let cursor_force_magnitude = settings.cursor_strength * falloff;
            
            total_force += cursor_delta * (cursor_force_magnitude / cursor_distance);
        }
    }
    
    forces[particle_idx] = total_force;
}

/// Updates particle positions and velocities
/// 
/// For each particle:
/// 1. Apply forces to velocity
/// 2. Apply friction
/// 3. Update position
/// 4. Handle world boundaries
@compute @workgroup_size(WORKGROUP_SIZE)
fn update_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    let particle_count = arrayLength(&particles);
    
    if (particle_idx >= particle_count) {
        return;
    }
    
    var particle = particles[particle_idx];
    let force = forces[particle_idx];
    
    // Update velocity with force and friction
    particle.velocity += force * settings.dt;
    particle.velocity *= pow(settings.friction, settings.dt * 60.0);
    
    // Update position
    particle.position += particle.velocity * settings.dt;
    
    // Handle boundaries
    if (settings.wrap_boundaries != 0u) {
        let world_min = -settings.world_size * 0.5;
        particle.position.x = world_min + ((particle.position.x - world_min) % settings.world_size + settings.world_size) % settings.world_size;
        particle.position.y = world_min + ((particle.position.y - world_min) % settings.world_size + settings.world_size) % settings.world_size;
    } else {
        let world_half = settings.world_size * 0.5;
        particle.position.x = clamp(particle.position.x, -world_half, world_half);
        particle.position.y = clamp(particle.position.y, -world_half, world_half);
    }
    
    particle.position.z = 0.0;
    
    particles[particle_idx] = particle;
}