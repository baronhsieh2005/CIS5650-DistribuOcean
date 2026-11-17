
export type Vec3 = readonly [number, number, number];

export interface CompressedBoid {
  radius: number;
  azimuth: number;
  polar: number;
  rotationX: number;
  rotationY: number;
  rotationZ: number;
}

export interface CompressedFlock {
  position: Vec3;
  radius: number;
  direction: Vec3;
  count: number;
  boids: CompressedBoid[];
}

export interface ParsedCudaPacket {
  flocks: CompressedFlock[];
  totalBoids: number;
  byteLength: number;
}

export interface ReconstructedBoid {
  position: Vec3;
  forward: Vec3;
}

export const NETWORK_FLOCK_BYTES = 32;
export const COMPRESSED_BOID_BYTES = 6;
const MAX_ROTATION_RAD = Math.PI / 12; 

export function parseCudaPacket(packet: ArrayBuffer | Uint8Array): ParsedCudaPacket {
  const bytes = packet instanceof Uint8Array ? packet : new Uint8Array(packet);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const flocks: CompressedFlock[] = [];
  let totalBoids = 0;
  let offset = 0;
  while (offset < bytes.byteLength) {
    const remaining = bytes.byteLength - offset;
    if (remaining < NETWORK_FLOCK_BYTES) {
      throw new RangeError(`Truncated CUDA flock header at byte ${offset}`);
    }
    const f32 = (relative: number): number => view.getFloat32(offset + relative, true);
    const position: Vec3 = [f32(0), f32(4), f32(8)];
    const radius = f32(12);
    const direction: Vec3 = [f32(16), f32(20), f32(24)];
    const count = view.getUint32(offset + 28, true);

    if (![...position, radius, ...direction].every(Number.isFinite) || radius < 0) {
      throw new RangeError(`Invalid CUDA flock metadata at byte ${offset}`);
    }
    // Verify count BEFORE multiplying, so malformed packets cannot allocate a huge array.
    if (count > Math.floor((remaining - NETWORK_FLOCK_BYTES) / COMPRESSED_BOID_BYTES)) {
      throw new RangeError(`Truncated CUDA boid payload at byte ${offset}`);
    }
    const payloadBytes = NETWORK_FLOCK_BYTES + count * COMPRESSED_BOID_BYTES;
    const chunkBytes = Math.ceil(payloadBytes / 4) * 4;
    if (chunkBytes > remaining) {
      throw new RangeError(`Missing CUDA flock alignment bytes at byte ${offset}`);
    }
    const boids: CompressedBoid[] = [];
    let boidOffset = offset + NETWORK_FLOCK_BYTES;
    for (let i = 0; i < count; i++) {
      boids.push({
        radius: view.getUint8(boidOffset),
        azimuth: view.getUint8(boidOffset + 1),
        polar: view.getUint8(boidOffset + 2),
        rotationX: view.getUint8(boidOffset + 3),
        rotationY: view.getUint8(boidOffset + 4),
        rotationZ: view.getUint8(boidOffset + 5),
      });
      boidOffset += COMPRESSED_BOID_BYTES;
    }
    flocks.push({ position, radius, direction, count, boids });
    totalBoids += count;
    offset += chunkBytes;
  }
  return { flocks, totalBoids, byteLength: bytes.byteLength };
}

function signedAngle(byte: number): number {
  const sign = (byte & 0x80) ? -1 : 1;
  return sign * (byte & 0x7f) / 127 * MAX_ROTATION_RAD;
}


export function reconstructBoid(flock: CompressedFlock, boid: CompressedBoid): ReconstructedBoid {
  const radialDistance = flock.radius * (boid.radius / 255);
  const azimuth = boid.azimuth / 255 * (2 * Math.PI);
  const polar = boid.polar / 255 * Math.PI;
  const offset: Vec3 = [
    radialDistance * Math.sin(polar) * Math.cos(azimuth),
    radialDistance * Math.sin(polar) * Math.sin(azimuth),
    radialDistance * Math.cos(polar),
  ];
  const position: Vec3 = [
    flock.position[0] + offset[0],
    flock.position[1] + offset[1],
    flock.position[2] + offset[2],
  ];
  const baseYaw = Math.atan2(flock.direction[1], flock.direction[0]);
  const basePitch = Math.asin(Math.max(-1, Math.min(1, flock.direction[2])));
  const yaw = baseYaw + signedAngle(boid.rotationX);
  const pitch = basePitch + signedAngle(boid.rotationY);
  const forward: Vec3 = [
    Math.cos(pitch) * Math.cos(yaw),
    Math.cos(pitch) * Math.sin(yaw),
    Math.sin(pitch),
  ];
  return { position, forward };
}
