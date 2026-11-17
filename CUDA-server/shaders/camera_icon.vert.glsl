#version 430

struct CameraData {
    mat4 proj_mat;
};

layout(std430, binding = 0) readonly buffer CameraBuffer {
    CameraData cameras[];
};

uniform int currCamera;

in vec4 Position;
in vec4 Velocity;
out vec4 vColor;

void main() {
    vColor = Velocity;
    gl_Position = cameras[currCamera].proj_mat * Position;
}
