import renderShaderSource from '../shaders/boids_render.wgsl?raw';
import { OrbitCamera } from '../stage/camera';

export class BoidRenderer {
  private readonly pipeline: GPURenderPipeline;
  private depthTexture: GPUTexture | undefined;
  private depthWidth = 0;
  private depthHeight = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly canvas: HTMLCanvasElement,
    private readonly camera: OrbitCamera,
    format: GPUTextureFormat,
  ) {
    const module=device.createShaderModule({label:'instanced boids renderer',code:renderShaderSource});
    this.pipeline=device.createRenderPipeline({
      label:'boids perspective instanced triangles',
      layout:device.createPipelineLayout({bindGroupLayouts:[camera.layout]}),
      vertex:{module,entryPoint:'vertex',buffers:[{
        arrayStride:32,stepMode:'instance',attributes:[
          {shaderLocation:0,offset:0,format:'float32x4'},
          {shaderLocation:1,offset:16,format:'float32x4'},
        ],
      }]},
      fragment:{module,entryPoint:'fragment',targets:[{format}]},
      primitive:{topology:'triangle-list',cullMode:'none'},
      depthStencil:{format:'depth24plus',depthWriteEnabled:true,depthCompare:'less'},
    });
  }

  private getDepthView(): GPUTextureView {
    if (!this.depthTexture || this.depthWidth !== this.canvas.width || this.depthHeight !== this.canvas.height) {
      this.depthTexture?.destroy();
      this.depthWidth=this.canvas.width; this.depthHeight=this.canvas.height;
      this.depthTexture=this.device.createTexture({
        label:'boids depth',size:[this.depthWidth,this.depthHeight],
        format:'depth24plus',usage:GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    return this.depthTexture.createView();
  }

  draw(encoder: GPUCommandEncoder, vertexBuffer: GPUBuffer, count: number): void {
    const pass=encoder.beginRenderPass({
      label:'boids draw',
      colorAttachments:[{
        view:this.context.getCurrentTexture().createView(),
        clearValue:{r:0.018,g:0.03,b:0.052,a:1},loadOp:'clear',storeOp:'store',
      }],
      depthStencilAttachment:{
        view:this.getDepthView(),depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store',
      },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0,this.camera.bindGroup);
    pass.setVertexBuffer(0,vertexBuffer);
    pass.draw(3,count);
    pass.end();
  }
}
