#pragma once

#include "glslUtility.hpp"
#include "utilityCore.hpp"
#include "camera.hpp"
#include "globalCamera.hpp"
#include <vector>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

//====================================
// GL Stuff
//====================================

GLuint positionLocation = 0;   // Match results from glslUtility::createProgram.
GLuint velocitiesLocation = 1; // Also see attribtueLocations below.
const char *attributeLocations[] = { "Position", "Velocity" };

GLuint boidVAO = 0;
GLuint boidVBO_positions = 0;
GLuint boidVBO_velocities = 0;
GLuint boidVBO_cameras = 0;
GLuint boidIBO = 0;
GLuint cameraIconVAO = 0;
GLuint cameraIconVBO_positions = 0;
GLuint cameraIconVBO_colors = 0;
GLuint displayImage;
GLuint program[2];

const unsigned int PROG_BOID = 0;
const unsigned int PROG_CAMERA_ICON = 1;
const unsigned int cameraIconVerticesPerCamera = 16;
GLsizei cameraIconVertexCount = 0;

const float fovy = (float) (PI / 4);
const float zNear = 0.10f;
const float zFar = 10.0f;
int width = 1280;
int height = 720;
int pointSize = 2;


bool leftMousePressed = false;
bool rightMousePressed = false;

int currCamera = 0;
bool useGlobalCamera = false;
std::vector<Camera> cameras = {
    Camera(-0.70f, 1.22f, 45.0f, glm::vec3(-75.0f, -30.0f, 0.0f)),
    Camera(0.70f, 1.10f, 45.0f, glm::vec3(75.0f, -30.0f, 0.0f)),
    Camera(2.10f, 1.45f, 45.0f, glm::vec3(0.0f, 75.0f, 0.0f)),
};
GlobalCamera globalCamera;
double lastX = 0.0;
double lastY = 0.0;

//====================================
// Main
//====================================

const char *projectName;

int main(int argc, char* argv[]);

//====================================
// Main loop
//====================================
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void updateCamera();
void runCUDA();

//====================================
// Setup/init Stuff
//====================================
bool init(int argc, char **argv);
void initVAO();
void initShaders(GLuint *program);
