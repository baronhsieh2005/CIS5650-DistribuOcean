#include "camera.hpp"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace {
constexpr float kFovy = 3.14159265358979323846f / 4.0f;
constexpr float kNearPlane = 0.10f;
constexpr float kFarPlane = 160.0f;
}

Camera::Camera(float phi, float theta, float zoom, glm::vec3 lookAt)
    : lookAt(lookAt), position(0.0f), projMat(1.0f) {
  position.x = zoom * std::sin(phi) * std::sin(theta);
  position.y = zoom * std::cos(phi) * std::sin(theta);
  position.z = zoom * std::cos(theta);
  position += lookAt;
}

void Camera::rotate(float deltaYaw, float deltaPitch) {
  const glm::vec3 worldUp(0.0f, 0.0f, 1.0f);
  glm::vec3 forward = getForward();
  forward = glm::vec3(glm::rotate(glm::mat4(1.0f), -deltaYaw, worldUp) * glm::vec4(forward, 0.0f));

  const glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
  const glm::vec3 pitchedForward = glm::vec3(
      glm::rotate(glm::mat4(1.0f), deltaPitch, right) * glm::vec4(forward, 0.0f));
  if (std::abs(glm::dot(pitchedForward, worldUp)) < 0.99f) {
    forward = pitchedForward;
  }
  lookAt = position + glm::normalize(forward);
}

void Camera::move(float forwardDistance, float rightDistance, float upDistance) {
  const glm::vec3 worldUp(0.0f, 0.0f, 1.0f);
  const glm::vec3 forward = getForward();
  const glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
  const glm::vec3 displacement = forward * forwardDistance + right * rightDistance + worldUp * upDistance;
  position += displacement;
  lookAt += displacement;
}

void Camera::update(float aspectRatio) {
  const glm::mat4 projection = glm::perspective(kFovy, aspectRatio, kNearPlane, kFarPlane);
  const glm::mat4 view = glm::lookAt(position, lookAt, glm::vec3(0.0f, 0.0f, 1.0f));
  projMat = projection * view;
}

const glm::mat4& Camera::getProjMat() const {
  return projMat;
}

const glm::vec3& Camera::getPosition() const {
  return position;
}

glm::vec3 Camera::getForward() const {
  return glm::normalize(lookAt - position);
}

