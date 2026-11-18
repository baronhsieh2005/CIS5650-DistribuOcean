export type Vec3 = readonly [number, number, number];

export const PACKED_UPDATE_BOID_BYTES = 6;
export const PACKED_NEW_BOID_BYTES = 6;
export const NEW_BOID_POLAR_ENCODING_BIT = 0x8000;
export const NEW_BOID_SLOT_MASK = 0x7fff;
export const UPDATE_BOID_FREE_COMPONENT = 0xffff;

export interface PacketCameraData {
  position: Vec3;
  inverseProjView: Float32Array;
  maxDepth: number;
}

export interface UpdatePacketHeader {
  sequence: number;
  byteLength: number;
}

export interface NewPacketHeader {
  sequence: number;
  byteLength: number;
  camera: PacketCameraData;
}

export interface NetworkPacket {
  update: {
    header: UpdatePacketHeader;
    bytes: Uint8Array;
  };
  newBoids: {
    header: NewPacketHeader;
    bytes: Uint8Array;
  };
}

export type NetworkPacketCallback = (packet: NetworkPacket) => void;

let networkPacketCallback: NetworkPacketCallback | undefined;

export function setNetworkPacketCallback(callback: NetworkPacketCallback): void {
  networkPacketCallback = callback;
}

function validatePayload(header: UpdatePacketHeader | NewPacketHeader, bytes: Uint8Array,
  recordSize: number, name: string): void {
  if (!Number.isSafeInteger(header.sequence) || header.sequence < 0) {
    throw new RangeError(`${name} packet sequence must be a non-negative integer`);
  }
  if (!Number.isSafeInteger(header.byteLength) || header.byteLength < 0 ||
      header.byteLength > bytes.byteLength || header.byteLength % recordSize !== 0) {
    throw new RangeError(`${name} packet byteLength must fit its packed ${recordSize}-byte records`);
  }
}

export function onNetworkPacketReceived(
  updateHeader: UpdatePacketHeader,
  updateBytes: Uint8Array,
  newHeader: NewPacketHeader,
  newBytes: Uint8Array,
): void {
  validatePayload(updateHeader, updateBytes, PACKED_UPDATE_BOID_BYTES, 'Update');
  validatePayload(newHeader, newBytes, PACKED_NEW_BOID_BYTES, 'New-boid');

  if (updateHeader.sequence !== newHeader.sequence) {
    throw new RangeError('Update and new-boid packet sequences must match');
  }
  if (newHeader.camera.inverseProjView.length !== 16 ||
      !Number.isFinite(newHeader.camera.maxDepth) || newHeader.camera.maxDepth <= 0 ||
      !newHeader.camera.position.every(Number.isFinite)) {
    throw new RangeError('New-boid packet camera data is invalid');
  }

  if (!networkPacketCallback) {
    throw new Error('Network packet callback has not been configured');
  }

  networkPacketCallback({
    update: {
      header: updateHeader,
      bytes: updateBytes.subarray(0, updateHeader.byteLength),
    },
    newBoids: {
      header: newHeader,
      bytes: newBytes.subarray(0, newHeader.byteLength),
    },
  });
}
