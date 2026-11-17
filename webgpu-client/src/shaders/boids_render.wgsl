struct Camera {
    viewProjection: mat4x4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
};
struct BoidVertex {
    @location(0) position: vec4<f32>,
    @location(1) velocity: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vertex(@builtin(vertex_index) vertexIndex: u32, boid: BoidVertex) -> VertexOutput {
    let triangle = array<vec2<f32>, 3>(
        vec2<f32>(1.6, 0.0),
        vec2<f32>(-1.0, 0.7),
        vec2<f32>(-1.0, -0.7)
    );
    let velocity = boid.velocity.xyz;
    let screenVelocity = vec2<f32>(dot(velocity, camera.right.xyz), dot(velocity, camera.up.xyz));
    let length2 = dot(screenVelocity, screenVelocity);
    let forward = select(vec2<f32>(1.0, 0.0), screenVelocity * inverseSqrt(max(length2, 0.000001)), length2 > 0.000001);
    let side = vec2<f32>(-forward.y, forward.x);
    let local = triangle[vertexIndex];
    let offset = forward * local.x + side * local.y;
    let worldPosition = boid.position.xyz + camera.right.xyz * offset.x + camera.up.xyz * offset.y;
    let clip = camera.viewProjection * vec4<f32>(worldPosition, 1.0);
    let color = clamp(abs(velocity) * 0.6 + vec3<f32>(0.28, 0.48, 0.67), vec3<f32>(0.1), vec3<f32>(1.0));
    return VertexOutput(clip, color);
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
