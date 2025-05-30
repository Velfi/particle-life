struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>,
    species: u32,
};

@group(0) @binding(0)
var<storage, read> particles: array<Particle>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

const POINT_SIZE: f32 = 2.0; // Smaller point size for zoomed out view

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let particle_index = vertex_index / 6u; // 6 vertices per particle (2 triangles)
    let vertex_offset = vertex_index % 6u;
    let particle = particles[particle_index];
    
    // Transform from window coordinates (0-2400, 0-1800) to NDC (-1 to 1)
    let ndc_x = (particle.position.x / 2400.0) * 2.0 - 1.0;
    let ndc_y = 1.0 - (particle.position.y / 1800.0) * 2.0; // Flip Y coordinate
    
    // Calculate point size in NDC space
    let point_size_ndc = POINT_SIZE / 1200.0; // 1200 is half of 2400 (window width)
    
    // Create a quad centered on the particle position
    var offset = vec2<f32>(0.0, 0.0);
    var uv = vec2<f32>(0.0, 0.0);
    
    switch (vertex_offset) {
        case 0u: { 
            offset = vec2<f32>(-point_size_ndc, -point_size_ndc);
            uv = vec2<f32>(-1.0, -1.0);
        }
        case 1u: { 
            offset = vec2<f32>(point_size_ndc, -point_size_ndc);
            uv = vec2<f32>(1.0, -1.0);
        }
        case 2u: { 
            offset = vec2<f32>(-point_size_ndc, point_size_ndc);
            uv = vec2<f32>(-1.0, 1.0);
        }
        case 3u: { 
            offset = vec2<f32>(-point_size_ndc, point_size_ndc);
            uv = vec2<f32>(-1.0, 1.0);
        }
        case 4u: { 
            offset = vec2<f32>(point_size_ndc, -point_size_ndc);
            uv = vec2<f32>(1.0, -1.0);
        }
        case 5u: { 
            offset = vec2<f32>(point_size_ndc, point_size_ndc);
            uv = vec2<f32>(1.0, 1.0);
        }
        default: {}
    }
    
    var output: VertexOutput;
    output.position = vec4<f32>(ndc_x + offset.x, ndc_y + offset.y, 0.0, 1.0);
    output.uv = uv;
    
    // Set color based on species using switch statement
    switch (particle.species) {
        case 0u: { output.color = vec4<f32>(1.0, 0.0, 0.0, 1.0); }    // Red
        case 1u: { output.color = vec4<f32>(0.0, 1.0, 0.0, 1.0); }    // Green
        case 2u: { output.color = vec4<f32>(0.0, 0.0, 1.0, 1.0); }    // Blue
        case 3u: { output.color = vec4<f32>(1.0, 1.0, 0.0, 1.0); }    // Yellow
        case 4u: { output.color = vec4<f32>(1.0, 0.0, 1.0, 1.0); }    // Magenta
        case 5u: { output.color = vec4<f32>(0.0, 1.0, 1.0, 1.0); }    // Cyan
        case 6u: { output.color = vec4<f32>(1.0, 0.5, 0.0, 1.0); }    // Orange
        case 7u: { output.color = vec4<f32>(0.5, 0.0, 1.0, 1.0); }    // Purple
        case 8u: { output.color = vec4<f32>(0.0, 0.5, 0.5, 1.0); }    // Teal
        default: { output.color = vec4<f32>(1.0, 1.0, 1.0, 1.0); }    // White (fallback)
    }
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance from center of the quad
    let dist = length(input.uv);
    
    // Create a circular shape by discarding fragments outside the circle
    if (dist > 1.0) {
        discard;
    }
    
    // Add a slight fade at the edges for smoother appearance
    let alpha = 1.0 - smoothstep(0.8, 1.0, dist);
    return vec4<f32>(input.color.rgb, input.color.a * alpha);
} 