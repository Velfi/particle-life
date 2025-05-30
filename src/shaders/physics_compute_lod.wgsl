// Enhanced physics compute shader with Level-of-Detail (LOD) system

struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    type_id: u32,
    lod_level: u32,
}

struct PhysicsSettings {
    dt: f32,
    rmax: f32,
    friction: f32,
    force_multiplier: f32,
    world_size: f32,
    wrap_boundaries: u32,
    matrix_size: u32,
    cursor_active: u32,
    cursor_position: vec3<f32>,
    cursor_size: f32,
    cursor_strength: f32,
    // LOD settings
    camera_position: vec3<f32>,
    camera_size: f32,
    lod_distance_0: f32,  // High detail distance
    lod_distance_1: f32,  // Medium detail distance
    lod_distance_2: f32,  // Low detail distance
    enable_frustum_culling: u32,
    frustum_left: f32,
    frustum_right: f32,
    frustum_top: f32,
    frustum_bottom: f32,
    _padding: vec2<f32>,
}

struct SpatialGridParams {
    cell_size: f32,
    grid_size: u32,
    world_min: f32,
    _padding: f32,
}

struct LodStats {
    lod0_count: atomic<u32>,
    lod1_count: atomic<u32>,
    lod2_count: atomic<u32>,
    culled_count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec3<f32>>;
@group(0) @binding(2) var<uniform> settings: PhysicsSettings;
@group(0) @binding(3) var<storage, read> interaction_matrix: array<f32>;
@group(0) @binding(4) var<storage, read_write> spatial_grid: array<u32>;
@group(0) @binding(5) var<storage, read_write> grid_counts: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> grid_params: SpatialGridParams;
@group(0) @binding(7) var<storage, read_write> lod_stats: LodStats;

const WORKGROUP_SIZE: u32 = 64u;
const MAX_PARTICLES_PER_CELL: u32 = 64u;

// LOD levels
const LOD_HIGH: u32 = 0u;      // Full physics, every frame
const LOD_MEDIUM: u32 = 1u;    // Half physics update rate
const LOD_LOW: u32 = 2u;       // Quarter physics update rate
const LOD_CULLED: u32 = 3u;    // No physics, outside frustum

fn get_cell_index(pos: vec3<f32>) -> u32 {
    let shifted_x = pos.x - grid_params.world_min;
    let shifted_y = pos.y - grid_params.world_min;
    
    let gx = u32(clamp(shifted_x / grid_params.cell_size, 0.0, f32(grid_params.grid_size - 1u)));
    let gy = u32(clamp(shifted_y / grid_params.cell_size, 0.0, f32(grid_params.grid_size - 1u)));
    
    return gy * grid_params.grid_size + gx;
}

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

fn is_in_frustum(pos: vec3<f32>) -> bool {
    if (settings.enable_frustum_culling == 0u) {
        return true;
    }
    
    return pos.x >= settings.frustum_left && 
           pos.x <= settings.frustum_right &&
           pos.y >= settings.frustum_bottom && 
           pos.y <= settings.frustum_top;
}

fn calculate_lod_level(particle_pos: vec3<f32>) -> u32 {
    // Check frustum culling first
    if (!is_in_frustum(particle_pos)) {
        return LOD_CULLED;
    }
    
    // Calculate distance from camera
    let camera_delta = particle_pos - settings.camera_position;
    let distance = length(camera_delta);
    
    // Scale distance by camera zoom level
    let scaled_distance = distance / settings.camera_size;
    
    if (scaled_distance < settings.lod_distance_0) {
        return LOD_HIGH;
    } else if (scaled_distance < settings.lod_distance_1) {
        return LOD_MEDIUM;
    } else if (scaled_distance < settings.lod_distance_2) {
        return LOD_LOW;
    } else {
        return LOD_CULLED;
    }
}

// Clear spatial grid and LOD stats
@compute @workgroup_size(WORKGROUP_SIZE)
fn clear_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_cells = grid_params.grid_size * grid_params.grid_size;
    
    if (index == 0u) {
        // Clear LOD stats on first thread
        atomicStore(&lod_stats.lod0_count, 0u);
        atomicStore(&lod_stats.lod1_count, 0u);
        atomicStore(&lod_stats.lod2_count, 0u);
        atomicStore(&lod_stats.culled_count, 0u);
    }
    
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

// Calculate LOD levels and build spatial grid
@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_lod_and_build_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    let particle_count = arrayLength(&particles);
    
    if (particle_idx >= particle_count) {
        return;
    }
    
    var particle = particles[particle_idx];
    
    // Calculate LOD level for this particle
    let lod_level = calculate_lod_level(particle.position);
    particle.lod_level = lod_level;
    particles[particle_idx] = particle;
    
    // Update LOD statistics
    switch (lod_level) {
        case LOD_HIGH: {
            atomicAdd(&lod_stats.lod0_count, 1u);
        }
        case LOD_MEDIUM: {
            atomicAdd(&lod_stats.lod1_count, 1u);
        }
        case LOD_LOW: {
            atomicAdd(&lod_stats.lod2_count, 1u);
        }
        case LOD_CULLED: {
            atomicAdd(&lod_stats.culled_count, 1u);
        }
        default: {}
    }
    
    // Only add non-culled particles to spatial grid
    if (lod_level != LOD_CULLED) {
        let cell_idx = get_cell_index(particle.position);
        
        // Atomically increment grid count and get slot
        let slot = atomicAdd(&grid_counts[cell_idx], 1u);
        
        if (slot < MAX_PARTICLES_PER_CELL) {
            let grid_entry_idx = cell_idx * MAX_PARTICLES_PER_CELL + slot;
            spatial_grid[grid_entry_idx] = particle_idx;
        }
    }
}

// Calculate forces with LOD-aware updates
@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_forces_lod(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    let particle_count = arrayLength(&particles);
    
    if (particle_idx >= particle_count) {
        return;
    }
    
    let particle_i = particles[particle_idx];
    
    // Skip culled particles
    if (particle_i.lod_level == LOD_CULLED) {
        forces[particle_idx] = vec3<f32>(0.0);
        return;
    }
    
    // LOD-based update frequency
    let frame_counter = u32(settings.dt * 10000.0) % 4u; // Simple frame counter approximation
    
    // Skip updates based on LOD level to reduce computation
    if (particle_i.lod_level == LOD_MEDIUM && frame_counter % 2u != 0u) {
        return; // Skip every other frame for medium LOD
    }
    if (particle_i.lod_level == LOD_LOW && frame_counter % 4u != 0u) {
        return; // Update only every 4th frame for low LOD
    }
    
    var total_force = vec3<f32>(0.0);
    
    let rmax_sq = settings.rmax * settings.rmax;
    let beta = 0.3;
    
    // LOD-adjusted interaction range
    var effective_rmax = settings.rmax;
    if (particle_i.lod_level == LOD_MEDIUM) {
        effective_rmax *= 0.7; // Reduce interaction range for medium LOD
    } else if (particle_i.lod_level == LOD_LOW) {
        effective_rmax *= 0.5; // Further reduce for low LOD
    }
    
    let effective_rmax_sq = effective_rmax * effective_rmax;
    
    // Get nearby particles from spatial grid
    let particle_cell = get_cell_index(particle_i.position);
    let grid_size = grid_params.grid_size;
    
    // LOD-based neighbor search extent
    var search_extent = 1i;
    if (particle_i.lod_level == LOD_LOW) {
        search_extent = 0i; // Only check same cell for low LOD
    }
    
    // Check neighboring cells
    for (var dy = -search_extent; dy <= search_extent; dy++) {
        for (var dx = -search_extent; dx <= search_extent; dx++) {
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
                
                // Skip interactions with culled particles
                if (particle_j.lod_level == LOD_CULLED) {
                    continue;
                }
                
                let delta = wrap_distance(particle_j.position - particle_i.position);
                let distance_sq = dot(delta, delta);
                
                if (distance_sq > 0.0 && distance_sq < effective_rmax_sq) {
                    let distance = sqrt(distance_sq);
                    let matrix_idx = particle_i.type_id * settings.matrix_size + particle_j.type_id;
                    let attraction = interaction_matrix[matrix_idx];
                    
                    var force_magnitude: f32;
                    if (distance < beta * effective_rmax) {
                        force_magnitude = (distance / (beta * effective_rmax) - 1.0) * settings.force_multiplier;
                    } else {
                        force_magnitude = attraction * (1.0 - abs(1.0 + beta - 2.0 * distance / effective_rmax) / (1.0 - beta)) * settings.force_multiplier;
                    }
                    
                    // LOD-based force scaling
                    var force_scale = 1.0;
                    if (particle_i.lod_level == LOD_MEDIUM) {
                        force_scale = 0.8;
                    } else if (particle_i.lod_level == LOD_LOW) {
                        force_scale = 0.6;
                    }
                    
                    total_force += delta * (force_magnitude * force_scale / distance);
                }
            }
        }
    }
    
    // Add cursor forces if active (only for high LOD particles)
    if (settings.cursor_active != 0u && particle_i.lod_level == LOD_HIGH) {
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

// Update positions and velocities with LOD awareness
@compute @workgroup_size(WORKGROUP_SIZE)
fn update_particles_lod(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    let particle_count = arrayLength(&particles);
    
    if (particle_idx >= particle_count) {
        return;
    }
    
    var particle = particles[particle_idx];
    
    // Skip culled particles
    if (particle.lod_level == LOD_CULLED) {
        return;
    }
    
    let force = forces[particle_idx];
    
    // LOD-based time scaling for different update rates
    var effective_dt = settings.dt;
    if (particle.lod_level == LOD_MEDIUM) {
        effective_dt *= 2.0; // Compensate for half update rate
    } else if (particle.lod_level == LOD_LOW) {
        effective_dt *= 4.0; // Compensate for quarter update rate
    }
    
    // Update velocity with force and friction
    particle.velocity += force * effective_dt;
    particle.velocity *= pow(settings.friction, effective_dt * 60.0);
    
    // Update position
    particle.position += particle.velocity * effective_dt;
    
    // Handle boundaries with improved wrapping
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