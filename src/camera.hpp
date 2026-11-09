#pragma once

#include <glm/glm.hpp>

class Camera {
public:
  Camera(float phi = -0.70f, float theta = 1.22f, float zoom = 4.0f,
         glm::vec3 lookAt = glm::vec3(0.0f));

  void rotate(float deltaYaw, float deltaPitch);
  void move(float forwardDistance, float rightDistance, float upDistance);
  void update(float aspectRatio);
  const glm::mat4& getProjMat() const;
  const glm::vec3& getPosition() const;
  glm::vec3 getForward() const;

private:
  glm::vec3 lookAt;
  glm::vec3 position;
  glm::mat4 projMat;
};
