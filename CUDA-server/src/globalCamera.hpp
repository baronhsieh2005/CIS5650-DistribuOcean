#pragma once

#include <glm/glm.hpp>

class GlobalCamera {
public:
  GlobalCamera(float phi = -0.70f, float theta = 1.22f, float zoom = 600.0f);

  void orbit(float deltaPhi, float deltaTheta);
  void changeZoom(float delta);
  void update(float aspectRatio);
  const glm::mat4& getProjMat() const;

private:
  float theta;
  float phi;
  float zoom;
  glm::vec3 position;
  glm::mat4 projMat;
};
