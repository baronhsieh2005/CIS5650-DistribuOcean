export function makeBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

export function sortSizeFor(count: number, blockSize = 512): number {
  let size = blockSize;
  while (size < count) size *= 2;
  return size;
}
