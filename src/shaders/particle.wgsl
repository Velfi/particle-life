struct UniformData {
    view_proj: mat4x4<f32>,
    time: f32,
    particle_size: f32,
    cam_top_left: vec2<f32>,
    wrap: u32,
    show_tiling: u32,
    world_size: f32,
    tile_fade_strength: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: UniformData;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) size: f32,
    @location(3) quad_vertex: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec2<f32>,
    @location(2) size: f32,
    @location(3) quad_pos: vec2<f32>,
    @location(4) tile_position: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Calculate the final particle size
    let particle_size = input.size * uniforms.particle_size * 0.01;
    
    // Offset the particle position by the quad vertex position scaled by size
    let offset = input.quad_vertex * particle_size;
    
    var final_position = input.position.xy;
    var tile_pos = vec2<f32>(0.0, 0.0); // Center tile by default
    
    // If tiling is enabled, offset position based on instance index
    if (uniforms.show_tiling != 0u) {
        // Calculate tile offset from instance index (0-8 for 3x3 grid)
        let tile_x = f32(i32(instance_index % 3u) - 1); // -1, 0, 1
        let tile_y = f32(i32(instance_index / 3u) - 1); // -1, 0, 1
        tile_pos = vec2<f32>(tile_x, tile_y);
        final_position = input.position.xy + tile_pos * uniforms.world_size;
    }
    
    let world_pos = vec4<f32>(final_position + offset, input.position.z, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    
    // Pass through color and other attributes
    out.color = input.color;
    out.world_pos = final_position;
    out.size = particle_size;
    out.quad_pos = input.quad_vertex;
    out.tile_position = tile_pos;
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance from center of quad to make circular particles
    let distance = length(input.quad_pos);
    
    // Discard fragments outside the circle (radius = 1.0)
    if distance > 1.0 {
        discard;
    }
    
    // Create a smooth edge for anti-aliasing
    let alpha_factor = 1.0 - smoothstep(0.8, 1.0, distance);
    let base_alpha = 0.9;
    
    // Apply slight time-based variation for visual effect
    let time_effect = sin(uniforms.time * 2.0) * 0.05 + 0.95;
    
    // Calculate tile fade based on distance from center (0, 0)
    var tile_fade = 1.0;
    if (uniforms.show_tiling != 0u) {
        // Calculate distance from center tile (0, 0)
        let tile_distance = length(input.tile_position);
        
        // Apply fade based on distance from center
        // Center tile (distance = 0): full opacity
        // Edge tiles (distance = 1): reduced opacity  
        // Corner tiles (distance = sqrt(2) ≈ 1.414): most reduced opacity
        let max_distance = 1.414; // sqrt(2) for corner tiles
        let fade_amount = smoothstep(0.0, max_distance, tile_distance) * uniforms.tile_fade_strength;
        tile_fade = 1.0 - fade_amount;
    }
    
    let final_alpha = input.color.a * base_alpha * alpha_factor * tile_fade;
    return vec4<f32>(input.color.rgb * time_effect, final_alpha);
}