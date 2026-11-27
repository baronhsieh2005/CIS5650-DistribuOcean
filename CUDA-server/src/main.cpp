/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 5650
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"
#include "kernel.h"
#include "glintVisualizer.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <thread>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/matrix_transform.hpp>

std::unique_ptr<std::thread> network_packet_create_thread;

/**
* C main function.
*/
int main(int argc, char* argv[]) {
  projectName = "5650 CUDA Intro: Boids";

  if (init(argc, argv)) {
    mainLoop();
    Boids::stopPhoneNetworkSendLoops();
    Boids::stopNetworkPacketCreationLoop();
    network_packet_create_thread->join();
    Boids::endSimulation();
    return 0;
  } else {
    return 1;
  }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

double simulationClockSeconds() {
#if VISUALIZE
  return glfwGetTime();
#else
  static const auto startTime = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::steady_clock::now() - startTime;
  return std::chrono::duration<double>(elapsed).count();
#endif
}

void initializeVirtualCameras() {
  // Match the three centers in kernGenerateFlockPositions.  Leave room between
  // cameras and the region boundaries so random directions still see boids.
  const glm::vec3 spawnCenters[] = {
    glm::vec3(-75.0f, -30.0f, 0.0f),
    glm::vec3(75.0f, -30.0f, 0.0f),
    glm::vec3(0.0f, 75.0f, 0.0f),
  };
  std::mt19937 random(VIRTUAL_CAMERA_RANDOM_SEED);
  std::uniform_real_distribution<float> positionDistribution(
    -VIRTUAL_CAMERA_POSITION_EXTENT, VIRTUAL_CAMERA_POSITION_EXTENT);
  std::uniform_real_distribution<float> azimuthDistribution(0.0f, 2.0f * PI);
  std::uniform_real_distribution<float> verticalDistribution(-1.0f, 1.0f);

  cameras.clear();
  cameras.reserve(NUM_VIRTUAL_CAMERAS);
  for (uint32_t cameraIndex = 0; cameraIndex < NUM_VIRTUAL_CAMERAS; ++cameraIndex) {
    const glm::vec3 position = spawnCenters[cameraIndex % 3] +
      glm::vec3(positionDistribution(random), positionDistribution(random), positionDistribution(random));
    const float azimuth = azimuthDistribution(random);
    const float vertical = verticalDistribution(random);
    const float horizontal = std::sqrt(1.0f - vertical * vertical);
    const glm::vec3 forward(horizontal * std::cos(azimuth), horizontal * std::sin(azimuth), vertical);

    // Camera's constructor places it at lookAt + sphericalOffset.  Pick a
    // one-unit offset opposite its desired forward direction, so the camera
    // starts exactly at the sampled position and looks in the sampled direction.
    const glm::vec3 offset = -forward;
    const float theta = std::acos(offset.z);
    const float phi = std::atan2(offset.x, offset.y);
    cameras.emplace_back(phi, theta, 1.0f, position + forward);
  }
}

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
  // Set window title to "Student Name: [SM 2.0] GPU Name"
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count) {
    std::cout
    << "Error: GPU device number is greater than the number of devices!"
    << " Perhaps a CUDA-capable GPU is not installed?"
    << std::endl;
    return false;
  }
  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  int major = deviceProp.major;
  int minor = deviceProp.minor;

  std::ostringstream ss;
  ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();

  network_packet_create_thread = std::make_unique<std::thread>(Boids::networkPacketCreationLoop);

  initializeVirtualCameras();

  // Rendering setup is not needed in compute-only mode.
#if VISUALIZE
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    std::cout
    << "Error: Could not initialize GLFW!"
    << " Perhaps OpenGL 3.3 isn't available?"
    << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    return false;
  }

  // Initialize drawing state
  initVAO();

  // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
  // change the device ID.
  cudaGLSetGLDevice(0);

  cudaGLRegisterBufferObject(boidVBO_positions);
  cudaGLRegisterBufferObject(boidVBO_velocities);
#endif

  // Initialize N-body simulation
  Boids::initSimulation(N_FOR_VIS);

#if VISUALIZE
  initShaders(program);
#endif
  updateAllCameras();

  networkCameras.resize(cameras.size());
  networkCameraChanged = std::make_unique<bool[]>(cameras.size());
  for (std::size_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex) {
    networkCameras[cameraIndex] = {
      cameras[cameraIndex].getPosition(),
      cameras[cameraIndex].getProjMat(),
    };
    networkCameraChanged[cameraIndex] = false;
  }
  Boids::initializeNetworkPhones(networkCameras.data(),
    static_cast<uint32_t>(networkCameras.size()));

#if VISUALIZE
  glEnable(GL_DEPTH_TEST);
#endif

  return true;
}

void initVAO() {

  std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
  std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

  glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
  glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

  for (int i = 0; i < N_FOR_VIS; i++) {
    bodies[4 * i + 0] = 0.0f;
    bodies[4 * i + 1] = 0.0f;
    bodies[4 * i + 2] = 0.0f;
    bodies[4 * i + 3] = 1.0f;
    bindices[i] = i;
  }


  glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
  glGenBuffers(1, &boidVBO_positions);
  glGenBuffers(1, &boidVBO_velocities);
  glGenBuffers(1, &boidVBO_cameras);
  glGenBuffers(1, &boidIBO);
  glGenVertexArrays(1, &cameraIconVAO);
  glGenBuffers(1, &cameraIconVBO_positions);
  glGenBuffers(1, &cameraIconVBO_colors);

  glBindVertexArray(boidVAO);

  // Bind the positions array to the boidVAO by way of the boidVBO_positions
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, boidVBO_cameras);
  cameraBufferEntryCapacity = cameras.size() + 1;
  glBufferData(GL_SHADER_STORAGE_BUFFER, cameraBufferEntryCapacity * sizeof(glm::mat4),
    nullptr, GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boidVBO_cameras);

  glBindVertexArray(cameraIconVAO);
  glBindBuffer(GL_ARRAY_BUFFER, cameraIconVBO_positions);
  glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer(positionLocation, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, cameraIconVBO_colors);
  glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer(velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

  glBindVertexArray(0);
}

void initShaders(GLuint * program) {
  GLint location;

  program[PROG_BOID] = glslUtility::createProgram(
    "shaders/boid.vert.glsl",
    "shaders/boid.geom.glsl",
    "shaders/boid.frag.glsl", attributeLocations, 2);
  program[PROG_CAMERA_ICON] = glslUtility::createProgram(
    "shaders/camera_icon.vert.glsl",
    "shaders/camera_icon.frag.glsl", attributeLocations, 2);

  }

  //====================================
  // Main loop
  //====================================
  void runCUDA() {
#if VISUALIZE
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer
    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
    cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);
#endif

#if SIMULATION_TYPE == GRID_SIMULATION
    Boids::stepSimulationCoherentGrid(DT, networkCameras.data(), networkCameraChanged.get(),
      static_cast<uint32_t>(networkCameras.size()));
#elif SIMULATION_TYPE == GLINT_SIMULATION
    Boids::stepSimulationGlint(DT, networkCameras.data(), networkCameraChanged.get(),
      static_cast<uint32_t>(networkCameras.size()));
#endif
    const double currentTime = simulationClockSeconds();
#if !VISUALIZE || DEBUG_MODE
    static double rateSampleStartTime = currentTime;
    static uint32_t stepsSinceRateSample = 0;
    ++stepsSinceRateSample;
#endif

#if !VISUALIZE || DEBUG_MODE
    const double rateSampleDuration = currentTime - rateSampleStartTime;
    if (rateSampleDuration >= 1.0) {
      printf("%.1f simulation steps/s", stepsSinceRateSample / rateSampleDuration);
#if DEBUG_MODE
      static uint32_t previouslyCompletedNetworkPackets = 0;
      const uint32_t completedPackets = Boids::completedNetworkPackets.load(std::memory_order_relaxed);
      const uint32_t packetsSinceRateSample = completedPackets - previouslyCompletedNetworkPackets;
      printf(" | %u packet builds completed (%.1f packets/s)", packetsSinceRateSample,
        packetsSinceRateSample / rateSampleDuration);
      previouslyCompletedNetworkPackets = completedPackets;
#endif
      printf("\n");
      rateSampleStartTime = currentTime;
      stepsSinceRateSample = 0;
    }
#endif

#if VISUALIZE
    Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
    // unmap buffer object
    cudaGLUnmapBufferObject(boidVBO_positions);
    cudaGLUnmapBufferObject(boidVBO_velocities);
#endif
  }

  void mainLoop() {
#if VISUALIZE
    double fps = 0;
    double timebase = 0;
    int frame = 0;


    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      frame++;
      double time = glfwGetTime();

      if (time - timebase > 1.0) {
        fps = frame / (time - timebase);
        timebase = time;
        frame = 0;
      }

      runCUDA();

      std::ostringstream ss;
      ss << "[";
      ss.precision(1);
      ss << std::fixed << fps;
      ss << " fps] " << deviceName;
      glfwSetWindowTitle(window, ss.str().c_str());

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      #if VISUALIZE
      glUseProgram(program[PROG_BOID]);
      glBindVertexArray(boidVAO);
      glPointSize((GLfloat)pointSize);
      glDrawElements(GL_POINTS, N_FOR_VIS, GL_UNSIGNED_INT, 0);
      glPointSize(1.0f);

      glUseProgram(program[PROG_CAMERA_ICON]);
      glBindVertexArray(cameraIconVAO);
      glLineWidth(3.0f);
      glDrawArrays(GL_LINES, 0, cameraIconVertexCount);
      glLineWidth(1.0f);

#if SIMULATION_TYPE == GLINT_SIMULATION
      GlintVisualizer::draw(program[PROG_CAMERA_ICON]);
#endif

      glUseProgram(0);
      glBindVertexArray(0);

      glfwSwapBuffers(window);
      #endif
    }
#if SIMULATION_TYPE == GLINT_SIMULATION
    GlintVisualizer::shutdown();
#endif
    glfwDestroyWindow(window);
    glfwTerminate();
#else
    while (true) {
      runCUDA();
    }
#endif
  }


  void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
    if (key == GLFW_KEY_F && action == GLFW_PRESS) {
      useGlobalCamera = false;
      currCamera = (currCamera + 1) % static_cast<int>(cameras.size());
      updateCameraIcons();
      updateActiveCameraUniform();
    }
    if (key == GLFW_KEY_G && action == GLFW_PRESS) {
      useGlobalCamera = !useGlobalCamera;
      updateCameraIcons();
      updateActiveCameraUniform();
    }
  #if VISUALIZE && SIMULATION_TYPE == GLINT_SIMULATION
    if (key == GLFW_KEY_H && action == GLFW_PRESS) {
      if (!(mods & GLFW_MOD_SHIFT)) {
        GlintVisualizer::toggle();
      }
    }
  #endif
    if (!useGlobalCamera && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
      constexpr float moveStep = 0.25f;
      if (key == GLFW_KEY_W) cameras[currCamera].move(moveStep, 0.0f, 0.0f);
      if (key == GLFW_KEY_S) cameras[currCamera].move(-moveStep, 0.0f, 0.0f);
      if (key == GLFW_KEY_A) cameras[currCamera].move(0.0f, -moveStep, 0.0f);
      if (key == GLFW_KEY_D) cameras[currCamera].move(0.0f, moveStep, 0.0f);
      if (key == GLFW_KEY_Q) cameras[currCamera].move(0.0f, 0.0f, -moveStep);
      if (key == GLFW_KEY_E) cameras[currCamera].move(0.0f, 0.0f, moveStep);
      if (key == GLFW_KEY_W || key == GLFW_KEY_S || key == GLFW_KEY_A ||
          key == GLFW_KEY_D || key == GLFW_KEY_Q || key == GLFW_KEY_E) {
        updateLocalCamera(currCamera);
        updateNetworkCamera(currCamera);
      }
    }
  }

  void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  }

  void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
      if (useGlobalCamera) {
        globalCamera.orbit((xpos - lastX) / width, -(ypos - lastY) / height);
      } else {
        cameras[currCamera].rotate((xpos - lastX) / width, -(ypos - lastY) / height);
      }
      if (!useGlobalCamera) {
        updateLocalCamera(currCamera);
        updateNetworkCamera(currCamera);
      } else {
        updateGlobalCamera();
      }
    } else if (rightMousePressed && useGlobalCamera) {
      globalCamera.changeZoom((ypos - lastY) / height * 100.0f);
      updateGlobalCamera();
    }

	lastX = xpos;
	lastY = ypos;
  }

  void updateCameraIcons() {
#if VISUALIZE
    std::vector<glm::vec4> cameraIconPositions;
    std::vector<glm::vec4> cameraIconColors;
    cameraIconPositions.reserve(cameras.size() * cameraIconVerticesPerCamera);
    cameraIconColors.reserve(cameras.size() * cameraIconVerticesPerCamera);
    for (std::size_t i = 0; i < cameras.size(); ++i) {
      if (!useGlobalCamera && static_cast<int>(i) == currCamera) {
        continue;
      }

      const glm::vec3 position = cameras[i].getPosition();
      const glm::vec3 forward = cameras[i].getForward();
      const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 0.0f, 1.0f)));
      const glm::vec3 up = glm::normalize(glm::cross(right, forward));
      const glm::vec3 tip = position + forward * 8.0f;
      const glm::vec3 topLeft = position - right * 4.0f + up * 3.0f;
      const glm::vec3 topRight = position + right * 4.0f + up * 3.0f;
      const glm::vec3 bottomRight = position + right * 4.0f - up * 3.0f;
      const glm::vec3 bottomLeft = position - right * 4.0f - up * 3.0f;
      const glm::vec4 iconColor(1.0f, 0.0f, 1.0f, 1.0f);
      const auto addIconEdge = [&](const glm::vec3& start, const glm::vec3& end) {
        cameraIconPositions.emplace_back(start, 1.0f);
        cameraIconPositions.emplace_back(end, 1.0f);
        cameraIconColors.push_back(iconColor);
        cameraIconColors.push_back(iconColor);
      };

      addIconEdge(topLeft, topRight);
      addIconEdge(topRight, bottomRight);
      addIconEdge(bottomRight, bottomLeft);
      addIconEdge(bottomLeft, topLeft);
      addIconEdge(topLeft, tip);
      addIconEdge(topRight, tip);
      addIconEdge(bottomRight, tip);
      addIconEdge(bottomLeft, tip);
    }
    cameraIconVertexCount = static_cast<GLsizei>(cameraIconPositions.size());

    glBindBuffer(GL_ARRAY_BUFFER, cameraIconVBO_positions);
    glBufferData(GL_ARRAY_BUFFER, cameraIconPositions.size() * sizeof(glm::vec4),
                 cameraIconPositions.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, cameraIconVBO_colors);
    glBufferData(GL_ARRAY_BUFFER, cameraIconColors.size() * sizeof(glm::vec4),
                 cameraIconColors.data(), GL_DYNAMIC_DRAW);
#endif
  }

  void updateActiveCameraUniform() {
#if VISUALIZE
    const int activeCameraIndex = useGlobalCamera ? static_cast<int>(cameras.size()) : currCamera;
    for (GLuint shaderProgram : program) {
      glUseProgram(shaderProgram);
      const GLint location = glGetUniformLocation(shaderProgram, "currCamera");
      if (location != -1) {
        glUniform1i(location, activeCameraIndex);
      }
    }
    glUseProgram(0);
#endif
  }

  void updateAllCameras() {
    const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    std::vector<glm::mat4> cameraMatrices(cameras.size() + 1);
    for (std::size_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex) {
      cameras[cameraIndex].update(aspectRatio);
      cameraMatrices[cameraIndex] = cameras[cameraIndex].getProjMat();
    }
    globalCamera.update(aspectRatio);
    cameraMatrices[cameras.size()] = globalCamera.getProjMat();

#if VISUALIZE
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, boidVBO_cameras);
    if (cameraMatrices.size() > cameraBufferEntryCapacity) {
      cameraBufferEntryCapacity = cameraMatrices.size();
      glBufferData(GL_SHADER_STORAGE_BUFFER, cameraBufferEntryCapacity * sizeof(glm::mat4),
        nullptr, GL_DYNAMIC_DRAW);
    }
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
      cameraMatrices.size() * sizeof(glm::mat4), cameraMatrices.data());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boidVBO_cameras);
#endif

    updateCameraIcons();
    updateActiveCameraUniform();
  }

  void updateLocalCamera(std::size_t cameraIndex) {
#if VISUALIZE
    if (cameras.size() + 1 > cameraBufferEntryCapacity) {
      updateAllCameras();
      return;
    }
#endif

    const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    cameras[cameraIndex].update(aspectRatio);
#if VISUALIZE
    const glm::mat4 &projMat = cameras[cameraIndex].getProjMat();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, boidVBO_cameras);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, cameraIndex * sizeof(glm::mat4),
      sizeof(glm::mat4), &projMat);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boidVBO_cameras);
#endif

    updateCameraIcons();
    updateActiveCameraUniform();
  }

  void updateGlobalCamera() {
#if VISUALIZE
    if (cameras.size() + 1 > cameraBufferEntryCapacity) {
      updateAllCameras();
      return;
    }
#endif

    const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    globalCamera.update(aspectRatio);
#if VISUALIZE
    const glm::mat4 &projMat = globalCamera.getProjMat();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, boidVBO_cameras);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, cameras.size() * sizeof(glm::mat4),
      sizeof(glm::mat4), &projMat);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boidVBO_cameras);
#endif

    updateActiveCameraUniform();
  }

  void updateNetworkCamera(std::size_t cameraIndex) {
    networkCameras[cameraIndex] = {
      cameras[cameraIndex].getPosition(),
      cameras[cameraIndex].getProjMat(),
    };
    networkCameraChanged[cameraIndex] = true;
  }
