#version 430

in vec4 Position;
in vec4 Velocity;
out vec4 vFragColorVs;

void main() {
    vFragColorVs = Velocity;
    gl_Position = Position;
}
