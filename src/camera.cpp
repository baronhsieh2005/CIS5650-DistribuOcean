#include "camera.hpp"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace {
constexpr float kMinTheta = 0.01f;
constexpr float kMaxTheta = 3.14f;
constexpr float kMinZoom = 0.1f;
constexpr float kMaxZoom = 5.0f;
constexpr float kFovy = 3.14159265358979323846f / 4.0f;
constexpr float kNearPlane = 0.10f;
constexpr float kFarPlane = 10.0f;
}

Camera::Camera(float phi, float theta, float zoom, glm::vec3 lookAt)
    : theta(theta), phi(phi), zoom(zoom), lookAt(lookAt), position(0.0f),
      projMat(1.0f) {}

void Camera::orbit(float deltaPhi, float deltaTheta) {
  phi += deltaPhi;
  theta = std::clamp(theta + deltaTheta, kMinTheta, kMaxTheta);
}

void Camera::changeZoom(float delta) {
  zoom = std::clamp(zoom + delta, kMinZoom, kMaxZoom);
}

void Camera::update(float aspectRatio) {
  position.x = zoom * std::sin(phi) * std::sin(theta);
  position.y = zoom * std::cos(phi) * std::sin(theta);
  position.z = zoom * std::cos(theta);
  position += lookAt;

  const glm::mat4 projection = glm::perspective(kFovy, aspectRatio, kNearPlane, kFarPlane);
  const glm::mat4 view = glm::lookAt(position, lookAt, glm::vec3(0.0f, 0.0f, 1.0f));
  projMat = projection * view;
}

const glm::mat4& Camera::getProjMat() const {
  return projMat;
}
