struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>,
    species: u32,
    mass: f32,           // Added mass for conservation
    parameters: vec4<f32>, // Local parameters for each particle
};

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

const BOUNDS: vec2<f32> = vec2<f32>(2400.0, 1800.0);
const INTERACTION_RADIUS: f32 = 150.0;
const DAMPING: f32 = 0.98;  // Increased damping for stability
const MIN_DISTANCE: f32 = 4.0;
const MAX_SPEED: f32 = 12.0;  // Increased max speed
const BROWNIAN_STRENGTH: f32 = 0.5;  // Increased Brownian motion
const TEMPERATURE: f32 = 0.2;        // Increased temperature
const NUM_SPECIES: u32 = 9u;

// Flow-Lenia specific constants
const MASS_CONSERVATION: f32 = 0.5;  // Reduced mass conservation for more movement
const PARAMETER_DIFFUSION: f32 = 0.2; // Increased parameter diffusion
const MIN_MASS: f32 = 0.5;           // Increased minimum mass
const MAX_MASS: f32 = 2.0;
const FORCE_SCALE: f32 = 2.0;        // Added force scaling factor

// Grid parameters
const GRID_CELL_SIZE: f32 = INTERACTION_RADIUS;
const GRID_WIDTH: u32 = u32(BOUNDS.x / GRID_CELL_SIZE) + 1u;
const GRID_HEIGHT: u32 = u32(BOUNDS.y / GRID_CELL_SIZE) + 1u;

// Shared memory for caching particles
var<workgroup> shared_particles: array<Particle, 64>;

// Improved pseudo-random number generator
fn random(seed: u32) -> f32 {
    var x = seed;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    x = x ^ (x << 7u);
    x = x ^ (x >> 3u);
    x = x ^ (x << 11u);
    return f32(x) / f32(0xFFFFFFFFu);
}

// Generate a random direction vector
fn random_direction(seed: u32) -> vec2<f32> {
    // Use different seeds for x and y to ensure independence
    let angle = random(seed) * 6.28318530718; // 2π
    return vec2<f32>(cos(angle), sin(angle));
}

fn get_interaction_strength(p1: Particle, p2: Particle) -> f32 {
    // Base interaction from species
    let base_interaction = get_interaction_strength_base(p1.species, p2.species);
    
    // Modify interaction based on local parameters
    let param_diff = p1.parameters - p2.parameters;
    let param_influence = dot(param_diff, param_diff) * PARAMETER_DIFFUSION;
    
    // Mass influence on interaction
    let mass_ratio = p1.mass / p2.mass;
    let mass_influence = (mass_ratio - 1.0) * 0.5;
    
    // Scale the interaction strength
    return (base_interaction + param_influence + mass_influence) * FORCE_SCALE;
}

fn get_interaction_strength_base(species1: u32, species2: u32) -> f32 {
    // Ensure species values are within valid range
    let s1 = species1 % NUM_SPECIES;
    let s2 = species2 % NUM_SPECIES;
    
    switch (s1) {
        case 0u: {
            switch (s2) {
                case 0u: { return 0.3; }
                case 1u: { return 0.7; }
                case 2u: { return -0.7; }
                case 3u: { return 0.4; }
                case 4u: { return -0.3; }
                case 5u: { return 0.5; }
                case 6u: { return -0.4; }
                case 7u: { return 0.3; }
                case 8u: { return -0.5; }
                default: { return 0.0; }
            }
        }
        case 1u: {
            switch (s2) {
                case 0u: { return 0.5; }
                case 1u: { return 0.0; }
                case 2u: { return 0.3; }
                case 3u: { return -0.5; }
                case 4u: { return 0.4; }
                case 5u: { return -0.2; }
                case 6u: { return 0.3; }
                case 7u: { return -0.4; }
                case 8u: { return 0.2; }
                default: { return 0.0; }
            }
        }
        case 2u: {
            switch (s2) {
                case 0u: { return -0.5; }
                case 1u: { return 0.3; }
                case 2u: { return 0.0; }
                case 3u: { return 0.5; }
                case 4u: { return -0.3; }
                case 5u: { return 0.2; }
                case 6u: { return 0.4; }
                case 7u: { return -0.2; }
                case 8u: { return 0.3; }
                default: { return 0.0; }
            }
        }
        case 3u: {
            switch (s2) {
                case 0u: { return 0.3; }
                case 1u: { return -0.5; }
                case 2u: { return 0.5; }
                case 3u: { return 0.0; }
                case 4u: { return 0.2; }
                case 5u: { return -0.3; }
                case 6u: { return 0.4; }
                case 7u: { return 0.3; }
                case 8u: { return -0.2; }
                default: { return 0.0; }
            }
        }
        case 4u: {
            switch (s2) {
                case 0u: { return -0.2; }
                case 1u: { return 0.4; }
                case 2u: { return -0.3; }
                case 3u: { return 0.2; }
                case 4u: { return 0.0; }
                case 5u: { return 0.5; }
                case 6u: { return -0.4; }
                case 7u: { return 0.3; }
                case 8u: { return 0.2; }
                default: { return 0.0; }
            }
        }
        case 5u: {
            switch (s2) {
                case 0u: { return 0.4; }
                case 1u: { return -0.2; }
                case 2u: { return 0.2; }
                case 3u: { return -0.3; }
                case 4u: { return 0.5; }
                case 5u: { return 0.0; }
                case 6u: { return 0.3; }
                case 7u: { return -0.4; }
                case 8u: { return 0.2; }
                default: { return 0.0; }
            }
        }
        case 6u: {
            switch (s2) {
                case 0u: { return -0.3; }
                case 1u: { return 0.3; }
                case 2u: { return 0.4; }
                case 3u: { return 0.4; }
                case 4u: { return -0.4; }
                case 5u: { return 0.3; }
                case 6u: { return 0.0; }
                case 7u: { return 0.5; }
                case 8u: { return -0.3; }
                default: { return 0.0; }
            }
        }
        case 7u: {
            switch (s2) {
                case 0u: { return 0.2; }
                case 1u: { return -0.4; }
                case 2u: { return -0.2; }
                case 3u: { return 0.3; }
                case 4u: { return 0.3; }
                case 5u: { return -0.4; }
                case 6u: { return 0.5; }
                case 7u: { return 0.0; }
                case 8u: { return 0.4; }
                default: { return 0.0; }
            }
        }
        case 8u: {
            switch (s2) {
                case 0u: { return -0.4; }
                case 1u: { return 0.2; }
                case 2u: { return 0.3; }
                case 3u: { return -0.2; }
                case 4u: { return 0.2; }
                case 5u: { return 0.2; }
                case 6u: { return -0.3; }
                case 7u: { return 0.4; }
                case 8u: { return 0.0; }
                default: { return 0.0; }
            }
        }
        default: { return 0.0; }
    }
}

fn get_grid_cell(pos: vec2<f32>) -> vec2<u32> {
    return vec2<u32>(
        u32(pos.x / GRID_CELL_SIZE),
        u32(pos.y / GRID_CELL_SIZE)
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let p1 = particles[i];
    var force = vec2<f32>(0.0, 0.0);
    var mass_change = 0.0;
    var parameter_change = vec4<f32>(0.0);
    
    // Get the grid cell of the current particle
    let cell = get_grid_cell(p1.position);
    
    // Check neighboring cells (including current cell)
    for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
        for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
            let neighbor_cell = vec2<i32>(i32(cell.x) + dx, i32(cell.y) + dy);
            
            // Skip if out of bounds
            if (neighbor_cell.x < 0i || neighbor_cell.x >= i32(GRID_WIDTH) ||
                neighbor_cell.y < 0i || neighbor_cell.y >= i32(GRID_HEIGHT)) {
                continue;
            }
            
            // Calculate cell index and load particles
            let cell_index = u32(neighbor_cell.y) * GRID_WIDTH + u32(neighbor_cell.x);
            let cell_start = cell_index * 64u;
            
            // Load particles into shared memory
            for (var j = 0u; j < 64u; j = j + 1u) {
                let particle_index = cell_start + j;
                if (particle_index < arrayLength(&particles)) {
                    shared_particles[j] = particles[particle_index];
                }
            }
            
            workgroupBarrier();
            
            // Process interactions
            for (var j = 0u; j < 64u; j = j + 1u) {
                let p2 = shared_particles[j];
                if (i == cell_start + j) { continue; }
                
                let diff = p2.position - p1.position;
                let dist = length(diff);
                
                if (dist < MIN_DISTANCE || dist > INTERACTION_RADIUS) { continue; }
                
                // Calculate interaction based on Flow-Lenia rules
                let interaction = get_interaction_strength(p1, p2);
                let force_magnitude = interaction * (1.0 - dist / INTERACTION_RADIUS);
                force = force + normalize(diff) * force_magnitude;
                
                // Mass transfer based on interaction
                let mass_transfer = interaction * (1.0 - dist / INTERACTION_RADIUS) * MASS_CONSERVATION;
                mass_change = mass_change + mass_transfer;
                
                // Parameter diffusion
                let param_diff = p2.parameters - p1.parameters;
                parameter_change = parameter_change + param_diff * PARAMETER_DIFFUSION * (1.0 - dist / INTERACTION_RADIUS);
            }
            
            workgroupBarrier();
        }
    }

    // Add Brownian motion with increased strength
    let seed = u32(i) + u32(global_id.y) * 1000u + u32(global_id.z) * 1000000u;
    let random_dir = random_direction(seed);
    let brownian_force = random_dir * BROWNIAN_STRENGTH * TEMPERATURE;
    force = force + brownian_force;

    // Update velocity with force and damping
    var new_velocity = p1.velocity * DAMPING + force;
    
    // Limit maximum velocity
    let speed = length(new_velocity);
    if (speed > MAX_SPEED) {
        new_velocity = normalize(new_velocity) * MAX_SPEED;
    }

    // Update position
    var new_position = p1.position + new_velocity;
    
    // Toroidal boundary conditions
    if (new_position.x < 0.0) { new_position.x = BOUNDS.x; }
    if (new_position.x > BOUNDS.x) { new_position.x = 0.0; }
    if (new_position.y < 0.0) { new_position.y = BOUNDS.y; }
    if (new_position.y > BOUNDS.y) { new_position.y = 0.0; }

    // Update mass with conservation
    var new_mass = p1.mass + mass_change;
    new_mass = clamp(new_mass, MIN_MASS, MAX_MASS);
    
    // Update parameters with diffusion
    var new_parameters = p1.parameters + parameter_change;
    new_parameters = clamp(new_parameters, vec4<f32>(0.0), vec4<f32>(1.0));

    // Update particle state
    particles[i].position = new_position;
    particles[i].velocity = new_velocity;
    particles[i].mass = new_mass;
    particles[i].parameters = new_parameters;
    particles[i].species = p1.species % NUM_SPECIES;
} 