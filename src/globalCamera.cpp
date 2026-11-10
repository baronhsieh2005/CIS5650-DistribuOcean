#include "globalCamera.hpp"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace {
constexpr float kFovy = 3.14159265358979323846f / 4.0f;
constexpr float kNearPlane = 0.10f;
constexpr float kFarPlane = 1000.0f;
}

GlobalCamera::GlobalCamera(float phi, float theta, float zoom)
    : theta(theta), phi(phi), zoom(zoom), position(0.0f), projMat(1.0f) {}

void GlobalCamera::orbit(float deltaPhi, float deltaTheta) {
  phi += deltaPhi;
  theta = std::clamp(theta + deltaTheta, 0.01f, 3.14f);
}

void GlobalCamera::changeZoom(float delta) {
  zoom = std::clamp(zoom + delta, 50.0f, 1000.0f);
}

void GlobalCamera::update(float aspectRatio) {
  position.x = zoom * std::sin(phi) * std::sin(theta);
  position.y = zoom * std::cos(phi) * std::sin(theta);
  position.z = zoom * std::cos(theta);

  const glm::mat4 projection = glm::perspective(kFovy, aspectRatio, kNearPlane, kFarPlane);
  const glm::mat4 view = glm::lookAt(position, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  projMat = projection * view;
}

const glm::mat4& GlobalCamera::getProjMat() const {
  return projMat;
}
