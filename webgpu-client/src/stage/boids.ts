import computeShaderSource from '../shaders/boids.wgsl?raw';
import {
  DEFAULT_BOIDS, GRID_CELLS, MAX_BOIDS, MAX_SCAN_BLOCK_COUNT, MAX_SORT_SIZE,
  RADIX_PASSES, SCAN_BLOCK_SIZE, WORKGROUP_SIZE,
} from '../config';
import { makeBuffer, sortSizeFor } from '../utils/gpu';

export class LocalBoids {
  readonly device: GPUDevice;
  private readonly settings: GPUBuffer;
  private readonly radixBits: GPUBuffer;
  private readonly boidsA: GPUBuffer;
  private readonly boidsB: GPUBuffer;
  private readonly reordered: GPUBuffer;
  private readonly pairsA: GPUBuffer;
  private readonly pairsB: GPUBuffer;
  private readonly bits: GPUBuffer;
  private readonly scan: GPUBuffer;
  private readonly cellStart: GPUBuffer;
  private readonly cellEnd: GPUBuffer;
  private readonly blockSums: GPUBuffer;
  private readonly computeGroups: [GPUBindGroup, GPUBindGroup];
  private readonly pipelines: Record<string, GPUComputePipeline>;
  private count = DEFAULT_BOIDS;
  private sortSize = sortSizeFor(DEFAULT_BOIDS);
  private activeA = true;

  constructor(device: GPUDevice) {
    this.device = device;
    const storage = GPUBufferUsage.STORAGE;
    const vertex = GPUBufferUsage.VERTEX;
    this.settings = makeBuffer(device, 'boid settings', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.radixBits = makeBuffer(device, 'radix bit uniforms', RADIX_PASSES * 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.boidsA = makeBuffer(device, 'boid state A', MAX_SORT_SIZE * 32, storage | vertex);
    this.boidsB = makeBuffer(device, 'boid state B', MAX_SORT_SIZE * 32, storage | vertex);
    this.reordered = makeBuffer(device, 'cell coherent state', MAX_SORT_SIZE * 32, storage);
    this.pairsA = makeBuffer(device, 'radix key/index A', MAX_SORT_SIZE * 8, storage);
    this.pairsB = makeBuffer(device, 'radix key/index B', MAX_SORT_SIZE * 8, storage);
    this.bits = makeBuffer(device, 'radix bits', MAX_SORT_SIZE * 4, storage);
    this.scan = makeBuffer(device, 'radix scans', MAX_SORT_SIZE * 4, storage);
    this.cellStart = makeBuffer(device, 'cell start', GRID_CELLS * 4, storage);
    this.cellEnd = makeBuffer(device, 'cell end', GRID_CELLS * 4, storage);
    this.blockSums = makeBuffer(device, 'radix scan block sums', MAX_SCAN_BLOCK_COUNT * 4, storage);

    const computeLayout = device.createBindGroupLayout({
      label: 'local boid compute layout',
      entries: [
        ...Array.from({ length: 10 }, (_, binding) => ({
          binding, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as const },
        })),
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 11, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 4 } },
      ],
    });
    this.computeGroups = [this.makeGroup(computeLayout, this.boidsA, this.boidsB), this.makeGroup(computeLayout, this.boidsB, this.boidsA)];
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [computeLayout] });
    const module = device.createShaderModule({ label: 'local boids compute', code: computeShaderSource });
    const entryPoints = [
      'initialize', 'assignPairs', 'radixABMap', 'radixBAMap',
      'scanBlocks', 'scanBlockSums', 'addBlockSums',
      'radixABScatter', 'radixBAScatter',
      'clearCells', 'buildCellRanges', 'gather', 'update',
    ];
    this.pipelines = Object.fromEntries(entryPoints.map(entryPoint => [
      entryPoint, device.createComputePipeline({
        label: `local boids ${entryPoint}`, layout: pipelineLayout,
        compute: { module, entryPoint },
      }),
    ]));
    for (let bit = 0; bit < RADIX_PASSES; bit++) {
      device.queue.writeBuffer(this.radixBits, bit * 256, new Uint32Array([bit]));
    }
    this.reset(DEFAULT_BOIDS);
  }

  private makeGroup(layout: GPUBindGroupLayout, current: GPUBuffer, next: GPUBuffer): GPUBindGroup {
    const buffers = [current, this.reordered, next, this.pairsA, this.pairsB,
      this.bits, this.scan, this.cellStart, this.cellEnd, this.blockSums,
      this.settings, this.radixBits];
    return this.device.createBindGroup({
      label: 'local boids compute group', layout,
      entries: buffers.map((buffer, binding) => ({
        binding, resource: binding === 11 ? { buffer, size: 16 } : { buffer },
      })),
    });
  }

  get boidCount(): number { return this.count; }

  reset(count: number): void {
    if (!Number.isInteger(count) || count < 1 || count > MAX_BOIDS) {
      throw new RangeError(`Boid count must be in [1, ${MAX_BOIDS}]`);
    }
    this.count = count;
    this.sortSize = sortSizeFor(count, SCAN_BLOCK_SIZE);
    this.activeA = true;
    this.device.queue.writeBuffer(this.settings, 0, new Uint32Array([count, this.sortSize]));
    const encoder = this.device.createCommandEncoder({ label: 'local boids reset' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.initialize);
    pass.setBindGroup(0, this.computeGroups[0], [0]);
    pass.dispatchWorkgroups(Math.ceil(this.sortSize / WORKGROUP_SIZE));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  simulate(encoder: GPUCommandEncoder): GPUBuffer {
    const group = this.computeGroups[this.activeA ? 0 : 1];
    const output = this.activeA ? this.boidsB : this.boidsA;
    const pass = encoder.beginComputePass({ label: 'local boids update' });
    const run = (entryPoint: string, groups: number, bit = 0): void => {
      pass.setPipeline(this.pipelines[entryPoint]);
      pass.setBindGroup(0, group, [bit * 256]);
      pass.dispatchWorkgroups(groups);
    };
    run('assignPairs', Math.ceil(this.sortSize / WORKGROUP_SIZE));
    for (let bit = 0; bit < RADIX_PASSES; bit++) {
      const direction = bit % 2 === 0 ? 'AB' : 'BA';
      run(`radix${direction}Map`, Math.ceil(this.sortSize / WORKGROUP_SIZE), bit);
      run('scanBlocks', this.sortSize / SCAN_BLOCK_SIZE, bit);
      run('scanBlockSums', 1, bit);
      run('addBlockSums', this.sortSize / SCAN_BLOCK_SIZE, bit);
      run(`radix${direction}Scatter`, Math.ceil(this.sortSize / WORKGROUP_SIZE), bit);
    }
    run('clearCells', Math.ceil(GRID_CELLS / WORKGROUP_SIZE));
    run('buildCellRanges', Math.ceil(this.count / WORKGROUP_SIZE));
    run('gather', Math.ceil(this.count / WORKGROUP_SIZE));
    run('update', Math.ceil(this.count / WORKGROUP_SIZE));
    pass.end();
    this.activeA = !this.activeA;
    return output;
  }
}
