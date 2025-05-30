struct UniformData {
    view_proj: mat4x4<f32>,
    time: f32,
    particle_size: f32,
    cam_top_left: vec2<f32>,
    wrap: u32,
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
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Calculate the final particle size
    let particle_size = input.size * uniforms.particle_size * 0.01;
    
    // Offset the particle position by the quad vertex position scaled by size
    let offset = input.quad_vertex * particle_size;
    let world_pos = vec4<f32>(input.position.xy + offset, input.position.z, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    
    // Pass through color and world position
    out.color = input.color;
    out.world_pos = input.position.xy;
    out.size = particle_size;
    out.quad_pos = input.quad_vertex; // Pass the quad position for circle calculation
    
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
    
    return vec4<f32>(input.color.rgb * time_effect, input.color.a * base_alpha * alpha_factor);
}