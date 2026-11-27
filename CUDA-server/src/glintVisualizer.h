#pragma once

#include "config.h"

#if VISUALIZE && SIMULATION_TYPE == GLINT_SIMULATION
namespace GlintVisualizer {
void toggle();
void draw(unsigned int shaderProgram);
void shutdown();
}
#endif
