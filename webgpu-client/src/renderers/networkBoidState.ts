import { NetworkPacket, PACKED_NEW_BOID_BYTES, PACKED_UPDATE_BOID_BYTES, } from '../network/cudaPacket';
import { makeBuffer } from '../utils/gpu';

const BOID_STATE_BYTES = 32;
const PACKET_UNIFORM_BYTES = 96;
const DEFAULT_WORKGROUP_SIZE = 256;

export interface NetworkPacketComputePipelines {
  copyNextToPrevious: GPUComputePipeline;
  applyUpdates: GPUComputePipeline;
  applyNewBoids: GPUComputePipeline;
}

export class NetworkBoidState {
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly pipelineLayout: GPUPipelineLayout;

  private readonly device: GPUDevice;
  private readonly previousStateBuffer: GPUBuffer;
  private readonly nextStateBuffer: GPUBuffer;
  private readonly updatePacketBuffer: GPUBuffer;
  private readonly newPacketBuffer: GPUBuffer;
  private readonly newBoidMapBuffer: GPUBuffer;
  private readonly packetUniformBuffer: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly slotCapacity: number;
  private readonly workgroupSize: number;
  private pipelines: NetworkPacketComputePipelines | undefined;
  private lastSequence = -1;

  constructor(device: GPUDevice, slotCapacity = 30_000, workgroupSize = DEFAULT_WORKGROUP_SIZE) {
    if (!Number.isInteger(slotCapacity) || slotCapacity <= 0 || slotCapacity > 0x7fff) {
      throw new RangeError('Network boid slot capacity must be in [1, 32767]');
    }
    this.device = device;
    this.slotCapacity = slotCapacity;
    this.workgroupSize = workgroupSize;

    const stateUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX |
      GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const packetUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.previousStateBuffer = makeBuffer(device, 'network previous state', slotCapacity * BOID_STATE_BYTES, stateUsage);
    this.nextStateBuffer = makeBuffer(device, 'network next state', slotCapacity * BOID_STATE_BYTES, stateUsage);
    this.updatePacketBuffer = makeBuffer(device, 'network packed updates', align4(slotCapacity * PACKED_UPDATE_BOID_BYTES), packetUsage);
    this.newPacketBuffer = makeBuffer(device, 'network packed new boids', align4(slotCapacity * PACKED_NEW_BOID_BYTES), packetUsage);
    this.newBoidMapBuffer = makeBuffer(device, 'network new boid slot map', slotCapacity * Uint32Array.BYTES_PER_ELEMENT, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.packetUniformBuffer = makeBuffer(device, 'network packet uniforms', PACKET_UNIFORM_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'network packet decode layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.bindGroup = device.createBindGroup({
      label: 'network packet decode bind group',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.previousStateBuffer } },
        { binding: 1, resource: { buffer: this.nextStateBuffer } },
        { binding: 2, resource: { buffer: this.updatePacketBuffer } },
        { binding: 3, resource: { buffer: this.newPacketBuffer } },
        { binding: 4, resource: { buffer: this.newBoidMapBuffer } },
        { binding: 5, resource: { buffer: this.packetUniformBuffer } },
      ],
    });
  }

  setComputePipelines(pipelines: NetworkPacketComputePipelines): void {
    this.pipelines = pipelines;
  }

  get previousState(): GPUBuffer { return this.previousStateBuffer; }
  get nextState(): GPUBuffer { return this.nextStateBuffer; }
  get newBoidMap(): GPUBuffer { return this.newBoidMapBuffer; }

  compute(packet: NetworkPacket): void {
    if (!this.pipelines) {
      throw new Error('Network packet compute pipelines have not been configured');
    }
    const sequence = packet.update.header.sequence;
    if (sequence <= this.lastSequence) {
      return;
    }

    const updateCount = packet.update.header.byteLength / PACKED_UPDATE_BOID_BYTES;
    const newCount = packet.newBoids.header.byteLength / PACKED_NEW_BOID_BYTES;
    if (updateCount > this.slotCapacity || newCount > this.slotCapacity) {
      throw new RangeError('Network packet exceeds the configured boid slot capacity');
    }

    writePaddedBytes(this.device, this.updatePacketBuffer, packet.update.bytes);
    writePaddedBytes(this.device, this.newPacketBuffer, packet.newBoids.bytes);
    this.device.queue.writeBuffer(this.packetUniformBuffer, 0, makePacketUniforms(this.slotCapacity, updateCount, newCount, packet));

    const encoder = this.device.createCommandEncoder({ label: 'network packet decode' });

    const copyPass = encoder.beginComputePass({ label: 'copy next packet state to previous state' });
    copyPass.setPipeline(this.pipelines.copyNextToPrevious);
    copyPass.setBindGroup(0, this.bindGroup);
    copyPass.dispatchWorkgroups(Math.ceil(this.slotCapacity / this.workgroupSize));
    copyPass.end();

    const updatePass = encoder.beginComputePass({ label: 'apply network boid updates' });
    updatePass.setPipeline(this.pipelines.applyUpdates);
    updatePass.setBindGroup(0, this.bindGroup);
    updatePass.dispatchWorkgroups(Math.ceil(updateCount / this.workgroupSize));
    updatePass.end();

    const newBoidPass = encoder.beginComputePass({ label: 'apply network new boids' });
    newBoidPass.setPipeline(this.pipelines.applyNewBoids);
    newBoidPass.setBindGroup(0, this.bindGroup);
    newBoidPass.dispatchWorkgroups(Math.ceil(newCount / this.workgroupSize));
    newBoidPass.end();

    this.device.queue.submit([encoder.finish()]);
    this.lastSequence = sequence;
  }
}

function align4(byteLength: number): number {
  return (byteLength + 3) & ~3;
}

function writePaddedBytes(device: GPUDevice, buffer: GPUBuffer, bytes: Uint8Array): void {
  const padded = new Uint8Array(align4(bytes.byteLength));
  padded.set(bytes);
  device.queue.writeBuffer(buffer, 0, padded);
}

function makePacketUniforms(slotCapacity: number, updateCount: number, newCount: number, packet: NetworkPacket): ArrayBuffer {
  const buffer = new ArrayBuffer(PACKET_UNIFORM_BYTES);
  const uniforms = new Float32Array(buffer);
  const words = new Uint32Array(buffer);
  words[0] = updateCount;
  words[1] = newCount;
  words[2] = slotCapacity;
  words[3] = packet.update.header.sequence;
  uniforms.set(packet.newBoids.header.camera.position, 4);
  uniforms[7] = packet.newBoids.header.camera.maxDepth;
  uniforms.set(packet.newBoids.header.camera.inverseProjView, 8);
  return buffer;
}
