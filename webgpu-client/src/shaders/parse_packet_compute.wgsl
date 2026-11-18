const NETWORK_COMPRESS_POS_SCALAR: f32 = 1.0 / 1024.0;
const UPDATE_BOID_FREE_COMPONENT: i32 = -1;
const NEW_BOID_POLAR_ENCODING_BIT: u32 = 0x8000u;
const NEW_BOID_SLOT_MASK: u32 = 0x7fffu;
const U8_MAX: f32 = 255.0;
const U16_MAX: f32 = 65535.0;
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

struct NetworkBoidState {
  positionAndActive: vec4<f32>,
  forwardOrVelocity: vec4<f32>,
};

struct UpdateBoidData {
  posDeltaX: i32,
  posDeltaY: i32,
  posDeltaZ: i32,
};

struct ViewPositionEncoding {
  xPos: u32,
  yPos: u32,
  depth: u32,
};

struct PolarPositionEncoding {
  azimuth: u32,
  elevation: u32,
  radialDepth: u32,
};

struct NewBoidData {
  modeAndSlot: u32,
  view: ViewPositionEncoding,
  polar: PolarPositionEncoding,
};

struct PacketUniforms {
  updateCount: u32,
  newCount: u32,
  slotCapacity: u32,
  sequence: u32,
  cameraPositionAndMaxDepth: vec4<f32>,
  inverseProjView: mat4x4<f32>,
};

@group(0) @binding(0) var<storage, read_write> previousState: array<NetworkBoidState>;
@group(0) @binding(1) var<storage, read_write> nextState: array<NetworkBoidState>;
@group(0) @binding(2) var<storage, read> packedUpdates: array<u32>;
@group(0) @binding(3) var<storage, read> packedNewBoids: array<u32>;
@group(0) @binding(4) var<storage, read_write> newBoidMap: array<u32>;
@group(0) @binding(5) var<uniform> packet: PacketUniforms;

fn loadUpdateByte(byteOffset: u32) -> u32 {
  let word = packedUpdates[byteOffset / 4u];
  return (word >> ((byteOffset % 4u) * 8u)) & 0xffu;
}

fn loadNewBoidByte(byteOffset: u32) -> u32 {
  let word = packedNewBoids[byteOffset / 4u];
  return (word >> ((byteOffset % 4u) * 8u)) & 0xffu;
}

fn loadUpdateU16(byteOffset: u32) -> u32 {
  return loadUpdateByte(byteOffset) | (loadUpdateByte(byteOffset + 1u) << 8u);
}

fn loadNewBoidU16(byteOffset: u32) -> u32 {
  return loadNewBoidByte(byteOffset) | (loadNewBoidByte(byteOffset + 1u) << 8u);
}

fn signExtendI16(value: u32) -> i32 {
  if ((value & 0x8000u) != 0u) {
    return i32(value) - 65536;
  }
  return i32(value);
}

fn loadUpdateBoidData(recordIndex: u32) -> UpdateBoidData {
  let byteOffset = recordIndex * 6u;
  return UpdateBoidData(
    signExtendI16(loadUpdateU16(byteOffset)),
    signExtendI16(loadUpdateU16(byteOffset + 2u)),
    signExtendI16(loadUpdateU16(byteOffset + 4u)),
  );
}

fn loadNewBoidData(recordIndex: u32) -> NewBoidData {
  let byteOffset = recordIndex * 6u;
  let modeAndSlot = loadNewBoidU16(byteOffset);
  let firstComponent = loadNewBoidByte(byteOffset + 2u);
  let secondComponent = loadNewBoidByte(byteOffset + 3u);
  let depth = loadNewBoidU16(byteOffset + 4u);
  return NewBoidData(
    modeAndSlot,
    ViewPositionEncoding(firstComponent, secondComponent, depth),
    PolarPositionEncoding(firstComponent, secondComponent, depth),
  );
}

fn decodedDepth(encodedDepth: u32) -> f32 {
  return f32(encodedDepth) * packet.cameraPositionAndMaxDepth.w / U16_MAX;
}

@compute @workgroup_size(256)
fn copyNextToPrevious(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= packet.slotCapacity) {
    return;
  }

  previousState[slot] = nextState[slot];
  newBoidMap[slot] = 0u;
}

@compute @workgroup_size(256)
fn applyUpdates(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= packet.updateCount) {
    return;
  }

  let updateData = loadUpdateBoidData(slot);
  if (updateData.posDeltaX == UPDATE_BOID_FREE_COMPONENT &&
      updateData.posDeltaY == UPDATE_BOID_FREE_COMPONENT &&
      updateData.posDeltaZ == UPDATE_BOID_FREE_COMPONENT) {
    nextState[slot].positionAndActive = vec4f(0.0);
    nextState[slot].forwardOrVelocity = vec4f(0.0);
    return;
  }

  let delta = vec3f(f32(updateData.posDeltaX), f32(updateData.posDeltaY),
    f32(updateData.posDeltaZ)) * NETWORK_COMPRESS_POS_SCALAR;
  nextState[slot].positionAndActive.xyz += delta;
  nextState[slot].positionAndActive.w = 1.0;
}

@compute @workgroup_size(256)
fn applyNewBoids(@builtin(global_invocation_id) gid: vec3<u32>) {
  let recordIndex = gid.x;
  if (recordIndex >= packet.newCount) {
    return;
  }

  let newBoid = loadNewBoidData(recordIndex);
  let slot = newBoid.modeAndSlot & NEW_BOID_SLOT_MASK;
  if (slot >= packet.slotCapacity) {
    return;
  }

  let cameraPosition = packet.cameraPositionAndMaxDepth.xyz;
  var position: vec3<f32>;
  if ((newBoid.modeAndSlot & NEW_BOID_POLAR_ENCODING_BIT) != 0u) {
    let azimuth = f32(newBoid.polar.azimuth) * TWO_PI / U8_MAX;
    let elevation = f32(newBoid.polar.elevation) * PI / U8_MAX - PI * 0.5;
    let cosElevation = cos(elevation);
    let direction = vec3<f32>(cosElevation * cos(azimuth),
      cosElevation * sin(azimuth), sin(elevation));
    position = cameraPosition + direction * decodedDepth(newBoid.polar.radialDepth);
  } else {
    let ndcX = f32(newBoid.view.xPos) * 2.0 / U8_MAX - 1.0;
    let ndcY = f32(newBoid.view.yPos) * 2.0 / U8_MAX - 1.0;
    let farPointHomogeneous = packet.inverseProjView * vec4<f32>(ndcX, ndcY, 1.0, 1.0);
    if (abs(farPointHomogeneous.w) < 1.0e-6) {
      return;
    }
    let farPoint = farPointHomogeneous.xyz / farPointHomogeneous.w;
    let ray = farPoint - cameraPosition;
    let rayLengthSquared = dot(ray, ray);
    if (rayLengthSquared < 1.0e-12) {
      return;
    }
    position = cameraPosition + ray * inverseSqrt(rayLengthSquared) * decodedDepth(newBoid.view.depth);
  }

  nextState[slot].positionAndActive = vec4f(position, 1.0);
  let movement = position - previousState[slot].positionAndActive.xyz;
  let movementLengthSquared = dot(movement, movement);
  let forward = select(vec3f(0.0), movement * inverseSqrt(movementLengthSquared),
    movementLengthSquared > 1.0e-12);
  nextState[slot].forwardOrVelocity = vec4f(forward, 0.0);
  newBoidMap[slot] = 1u;
}
