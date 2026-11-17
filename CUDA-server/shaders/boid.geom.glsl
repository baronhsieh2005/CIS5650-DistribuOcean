#version 430

struct CameraData {
    mat4 proj_mat;
};

layout(std430, binding = 0) readonly buffer CameraBuffer {
    CameraData cameras[];
};

uniform int currCamera;

layout(points) in;
layout(points) out;
layout(max_vertices = 1) out;

in vec4 vFragColorVs[];
out vec4 vFragColor;

void main() {
    vec3 Position = gl_in[0].gl_Position.xyz;
    vFragColor = vFragColorVs[0];
    gl_Position = cameras[currCamera].proj_mat * vec4(Position, 1.0);
    EmitVertex();
    EndPrimitive();
}
