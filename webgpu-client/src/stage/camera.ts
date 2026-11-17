type Vec3 = [number, number, number];
function dot(a: Vec3, b: Vec3): number { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function normalize(a: Vec3): Vec3 {
  const n = Math.hypot(...a);
  return n > 1e-9 ? [a[0]/n,a[1]/n,a[2]/n] : [1,0,0];
}
function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
  const c = new Float32Array(16);
  for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) {
    for (let k = 0; k < 4; k++) c[j*4+i] += a[k*4+i]*b[j*4+k];
  }
  return c;
}
function viewProjection(yaw: number, pitch: number, distance: number, aspect: number): {
  matrix: Float32Array, right: Vec3, up: Vec3
} {
  const eye: Vec3 = [distance*Math.cos(pitch)*Math.cos(yaw),
    distance*Math.cos(pitch)*Math.sin(yaw), distance*Math.sin(pitch)];
  const forward = normalize([-eye[0], -eye[1], -eye[2]]);
  const right = normalize(cross(forward, [0,0,1]));
  const up = normalize(cross(right, forward));
  const view = new Float32Array([
    right[0], up[0], -forward[0], 0,
    right[1], up[1], -forward[1], 0,
    right[2], up[2], -forward[2], 0,
    -dot(right,eye), -dot(up,eye), dot(forward,eye), 1,
  ]);
  const fovY = Math.PI / 3, near = 0.1, far = 2000;
  const f = 1 / Math.tan(fovY/2);
  const projection = new Float32Array([
    f/aspect,0,0,0,
    0,f,0,0,
    0,0,far/(near-far),-1,
    0,0,(far*near)/(near-far),0,
  ]);
  return { matrix: multiplyMat4(projection, view), right, up };
}

export class OrbitCamera {
  readonly buffer: GPUBuffer;
  readonly layout: GPUBindGroupLayout;
  readonly bindGroup: GPUBindGroup;
  private yaw = 0.4;
  private pitch = 0.4;
  private distance = 380;
  private dragging: number | null = null;
  private x = 0;
  private y = 0;
  private lastPinchDistance = 0;

  constructor(private readonly device: GPUDevice, private readonly canvas: HTMLCanvasElement) {
    this.buffer = device.createBuffer({
      label: 'camera viewProjection/right/up', size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.layout = device.createBindGroupLayout({
      label: 'camera uniform layout',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    });
    this.bindGroup = device.createBindGroup({
      label: 'camera uniforms', layout: this.layout,
      entries: [{ binding: 0, resource: { buffer: this.buffer } }],
    });
    this.attachControls();
    this.update();
  }

  resize(): boolean {
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.min(this.device.limits.maxTextureDimension2D,
      Math.round(this.canvas.clientWidth * pixelRatio)));
    const height = Math.max(1, Math.min(this.device.limits.maxTextureDimension2D,
      Math.round(this.canvas.clientHeight * pixelRatio)));
    if (this.canvas.width === width && this.canvas.height === height) return false;
    this.canvas.width = width;
    this.canvas.height = height;
    this.update();
    return true;
  }

  private attachControls(): void {
    this.canvas.addEventListener('contextmenu', e => e.preventDefault());
    this.canvas.addEventListener('mousedown', e => {
      if (e.button !== 0 && e.button !== 2) return;
      this.dragging = e.button; this.x = e.clientX; this.y = e.clientY;
      this.canvas.style.cursor = 'grabbing';
    });
    window.addEventListener('mouseup', () => {
      this.dragging = null; this.canvas.style.cursor = 'grab';
    });
    window.addEventListener('mousemove', e => {
      if (this.dragging === null) return;
      const dx = (e.clientX-this.x) / Math.max(1,this.canvas.clientWidth);
      const dy = (e.clientY-this.y) / Math.max(1,this.canvas.clientHeight);
      if (this.dragging === 0) this.orbit(dx,dy);
      else this.zoomBy(dy * 300);
      this.x=e.clientX; this.y=e.clientY;
    });
    this.canvas.addEventListener('wheel', e => {
      e.preventDefault(); this.zoomBy(e.deltaY * 0.4);
    }, { passive: false });
    this.canvas.addEventListener('touchstart', e => {
      e.preventDefault();
      if (e.touches.length===1) {this.x=e.touches[0].clientX;this.y=e.touches[0].clientY;}
      if (e.touches.length===2) this.lastPinchDistance=Math.hypot(
        e.touches[0].clientX-e.touches[1].clientX, e.touches[0].clientY-e.touches[1].clientY);
    },{passive:false});
    this.canvas.addEventListener('touchmove',e=>{
      e.preventDefault();
      if(e.touches.length===1){
        const dx=(e.touches[0].clientX-this.x)/Math.max(1,this.canvas.clientWidth);
        const dy=(e.touches[0].clientY-this.y)/Math.max(1,this.canvas.clientHeight);
        this.orbit(dx,dy); this.x=e.touches[0].clientX; this.y=e.touches[0].clientY;
      }else if(e.touches.length===2){
        const distance=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,
          e.touches[0].clientY-e.touches[1].clientY);
        if(this.lastPinchDistance!==0)this.zoomBy((this.lastPinchDistance-distance)*1.2);
        this.lastPinchDistance=distance;
      }
    },{passive:false});
  }

  private orbit(dx: number, dy: number): void {
    this.yaw -= dx*3;
    this.pitch = Math.max(-Math.PI/2+0.04, Math.min(Math.PI/2-0.04, this.pitch+dy*3));
    this.update();
  }
  private zoomBy(amount: number): void {
    this.distance=Math.max(80,Math.min(1200,this.distance+amount));
    this.update();
  }
  private update(): void {
    const aspect=Math.max(1,this.canvas.width)/Math.max(1,this.canvas.height);
    const {matrix,right,up}=viewProjection(this.yaw,this.pitch,this.distance,aspect);
    const uniform=new Float32Array(24);
    uniform.set(matrix,0);
    uniform.set([...right,0],16);
    uniform.set([...up,0],20);
    this.device.queue.writeBuffer(this.buffer,0,uniform);
  }
}
