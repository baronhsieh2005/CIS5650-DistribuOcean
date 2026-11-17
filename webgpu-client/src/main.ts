import { GUI } from 'dat.gui';
import { DEFAULT_BOIDS, MAX_BOIDS } from './config';
import { LocalBoids } from './stage/boids';
import { OrbitCamera } from './stage/camera';
import { BoidRenderer } from './renderers/boidRenderer';

async function main(): Promise<void> {
  const canvas=document.getElementById('mainCanvas') as HTMLCanvasElement;
  const fpsCounter=document.getElementById('fpsCounter')!;
  const boidCounter=document.getElementById('boidCounter')!;
  if(!navigator.gpu) throw new Error('WebGPU unavailable on this browser');
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter) throw new Error('No WebGPU adapter available');
  if(adapter.limits.maxStorageBuffersPerShaderStage<10) {
    throw new Error(`Local simulation needs 10 storage buffers/stage; this adapter supports ${adapter.limits.maxStorageBuffersPerShaderStage}. `+
      'A phone-oriented CUDA streaming renderer can eliminate that requirement.');
  }
  const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:10}});
  device.addEventListener('uncapturederror',e=>console.error('WebGPU:',e.error));
  const context=canvas.getContext('webgpu');
  if(!context) throw new Error('Unable to obtain a WebGPU canvas context');
  const format=navigator.gpu.getPreferredCanvasFormat();
  context.configure({device,format,alphaMode:'opaque'});
  const camera=new OrbitCamera(device,canvas);
  const simulation=new LocalBoids(device);
  const renderer=new BoidRenderer(device,context,canvas,camera,format);
  const gui=new GUI();
  const controls={boidCount:DEFAULT_BOIDS};
  gui.add(controls,'boidCount',1,MAX_BOIDS,1).name('Boid count')
    .onFinishChange((count:number)=>simulation.reset(count));
  let frames=0,measurementStart=performance.now();
  function animate(now:number):void {
    camera.resize();
    const encoder=device.createCommandEncoder({label:'boid frame'});
    const output=simulation.simulate(encoder);
    renderer.draw(encoder,output,simulation.boidCount);
    device.queue.submit([encoder.finish()]);
    frames++;
    const elapsed=now-measurementStart;
    if(elapsed>=500){
      fpsCounter.textContent=`${(frames*1000/elapsed).toFixed(1)} FPS`;
      boidCounter.textContent=`${simulation.boidCount.toLocaleString()} boids`;
      frames=0;measurementStart=now;
    }
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}

main().catch(error=>{
  console.error(error);
  const message=document.createElement('p');
  message.className='error';
  message.textContent=error instanceof Error?error.message:String(error);
  document.body.appendChild(message);
});
