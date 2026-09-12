// CPU-only scene and event checks; no browser or GPU required. node tests/test_brain.cjs
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
(async()=>{
  const core=await import('../web/assets/three.core.js');
  const bodyContext={THREE:core};
  vm.runInNewContext(fs.readFileSync('web/fly-body.js','utf8').replace(/^import .*;$/mg,'').replace('export function','function'),bodyContext);
  const data=JSON.parse(fs.readFileSync('web/assets/fly-scene.json','utf8')),objects=bodyContext.flyObjects(data);
  assert(objects.find(o=>o.name==='nmf/l_eye').material.flatShading);
  assert(objects.find(o=>o.name==='nmf/l_wing').material.opacity<.5);
  assert(objects.reduce((n,o)=>n+o.children.filter(c=>c.name==='bristles').reduce((sum,c)=>sum+c.count,0),0)>500);
  assert.equal(objects.length,data.geoms.length);objects.forEach((o,i)=>assert.deepEqual(o.position.toArray(),data.positions[i]));
  const listeners={},elements={},draws=[],motion={matches:false};let renderLoop,scene,camera,now=1000;
  function element(id){return elements[id]??=(id==='canvas2d'?{getContext:()=>({fillRect(){},fillText(){},drawImage(...args){draws.push(args);}})}:{clientWidth:900,clientHeight:560,hidden:false,classList:{add(){},remove(){}},append(){},addEventListener(){}});}
  const context={THREE:{...core,TextureLoader:class{load(src){return new core.Texture({src});}},WebGLRenderer:class{setPixelRatio(){}setClearColor(){}setSize(){}constructor(){this.domElement=element('webgl');}setAnimationLoop(fn){renderLoop=fn;}render(s,c){scene=s;camera=c;}}},
    OrbitControls:class{constructor(c){this.camera=c;this.target=new core.Vector3();}update(){this.camera.lookAt(this.target);this.camera.updateMatrixWorld();}},
    flyObjects:bodyContext.flyObjects,document:{getElementById:element,createElement:()=>element('canvas2d'),hidden:false},
    window:{addEventListener:(type,fn)=>listeners[type]=fn},devicePixelRatio:1,matchMedia:()=>motion,performance:{now:()=>now},
    ResizeObserver:class{constructor(fn){this.fn=fn;}observe(){this.fn();}},fetch:async url=>({ok:true,json:async()=>url.includes('grooming')?JSON.parse(fs.readFileSync('web/assets/grooming.json','utf8')):data}),
    Image:class{set src(value){this.value=value;queueMicrotask(()=>this.onload());}},console};
  vm.runInNewContext(fs.readFileSync('web/brain.js','utf8').replace(/^import .*;$/mg,''),context);
  await new Promise(r=>setImmediate(r));
  renderLoop(now);assert(scene&&camera);assert(scene.getObjectByName('nmf/c_head'));
  const fly=scene.getObjectByName('diffusion-fly'),canvas=scene.getObjectByName('diffusion-screen'),cap=scene.getObjectByName('brain-cap');
  assert(cap&&cap.parent===fly,'Cap follows the fly');
  scene.updateMatrixWorld(true);
  const wordmark=scene.getObjectByName('screen-wordmark'),image=scene.getObjectByName('diffusion-image');
  assert.equal(wordmark.material.map.image.src,'/assets/flycasso-wordmark.png');
  assert(wordmark.position.z-wordmark.geometry.parameters.height/2>image.position.z+image.geometry.parameters.height/2,'Logo is above the entire generated image');
  const logoScreen=wordmark.getWorldPosition(new core.Vector3()).project(camera);assert(Math.abs(logoScreen.x)<1&&Math.abs(logoScreen.y)<1,'Screen wordmark fits the default camera');
  const forward=new core.Vector3(1,0,0).applyQuaternion(fly.quaternion),toward=canvas.position.clone().sub(fly.position).normalize();
  assert(forward.dot(toward)>.999,'Fly faces the screen');
  const normal=new core.Vector3(0,-1,0).applyQuaternion(canvas.quaternion);
  assert(normal.dot(toward)<-.999,'Image surface faces the fly, not its edge');
  for(const point of [cap.getWorldPosition(new core.Vector3()),new core.Vector3(2.4,0,1.48),new core.Vector3(-3.2,-1.5,1)]){
    const screen=point.clone().project(camera);assert(Math.abs(screen.x)<1&&Math.abs(screen.y)<1,'Fly, cap and canvas fit the camera');
  }
  const foot=fly.getObjectByName('nmf/lf_tarsus5'),rest=foot.position.clone(),support=fly.getObjectByName('nmf/lh_tarsus5'),supportRest=support.position.clone();
  listeners['diffusion-reset']();
  for(let t=now;t<=now+700;t+=16)renderLoop(t);
  assert(foot.position.distanceTo(rest)>.3,'Front foot lifts for rubbing');
  assert(foot.position.distanceTo(fly.getObjectByName('nmf/rf_tarsus5').position)<.25,'Front feet meet');
  assert.equal(support.position.distanceTo(supportRest),0,'Supporting legs remain still');
  const rubbing=foot.position.clone();renderLoop(now+740);assert(foot.position.distanceTo(rubbing)>.005,'Feet slide while rubbing');
  listeners['diffusion-end']();for(let t=now+750;t<now+4300;t+=16)renderLoop(t);
  assert(foot.position.distanceTo(rest)<1e-9,'Completion, failure or stop returns feet to rest');
  motion.matches=true;listeners['diffusion-reset']();for(let t=now+4400;t<now+4700;t+=16)renderLoop(t);
  assert.equal(foot.position.distanceTo(rest),0,'Reduced motion suppresses grooming');motion.matches=false;listeners['diffusion-end']();
  const pulse=scene.children.find(o=>o.geometry?.type==='SphereGeometry'&&o.material.type==='MeshBasicMaterial');
  assert.equal(pulse.visible,false);
  await listeners['diffusion-frame']({detail:{images:['actual-frame'],step:1}});renderLoop(now+450);assert.equal(pulse.visible,true);assert.equal(draws.at(-1)[0].value,'actual-frame');
  renderLoop(now+1000);assert.equal(pulse.visible,false,'Pulses stop without new model steps');
  listeners['diffusion-reset']();renderLoop(now);assert.equal(pulse.visible,false);
  const clip=JSON.parse(fs.readFileSync('web/assets/grooming.json','utf8'));
  assert(clip.minimum_clearance_mm>=-1e-5&&clip.collision_pairs>=80,'Baked poses pass mesh clearance checks');
  console.log('Facing direction, canvas orientation, joint grooming, stop/reduced-motion and live pulse checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
