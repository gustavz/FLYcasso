// A visual metaphor driven by received diffusion frames, never claimed as neural telemetry.
import * as THREE from 'three';
import {OrbitControls} from '/assets/OrbitControls.js';
import {flyObjects} from '/fly-body.js';

const $=id=>document.getElementById(id),view=$('brain-view');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#eee8e2');view.append(renderer.domElement);
const scene=new THREE.Scene(),camera=new THREE.OrthographicCamera(-4,4,3,-3,.01,50);
camera.up.set(0,0,1);
const controls=new OrbitControls(camera,renderer.domElement);controls.minZoom=.7;controls.maxZoom=3;controls.enablePan=false;
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
function resetCamera(){camera.position.set(-4,-8,5.4);camera.zoom=1;controls.target.set(.1,-.6,1.05);camera.updateProjectionMatrix();controls.update();}
$('brain-reset').onclick=resetCamera;resetCamera();
scene.add(new THREE.HemisphereLight(0xffffff,0x5c6151,3));
const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(-1,-3,7);scene.add(light);
const chrome=new THREE.MeshStandardMaterial({color:'#bcccca',metalness:.75,roughness:.3});
const dark=new THREE.MeshStandardMaterial({color:'#243c39',metalness:.35,roughness:.6});
const signal=new THREE.MeshStandardMaterial({color:'#80eee3',emissive:'#38cfbf',emissiveIntensity:.2,roughness:.3});
function mesh(geometry,material,position,parent=scene){const m=new THREE.Mesh(geometry,material);m.position.set(...position);parent.add(m);return m;}

// A single texture holds the same one or four images available in the download gallery.
const paper=document.createElement('canvas');paper.width=paper.height=512;
const ctx=paper.getContext('2d'),texture=new THREE.CanvasTexture(paper);
texture.colorSpace=THREE.SRGBColorSpace;texture.magFilter=THREE.NearestFilter;texture.minFilter=THREE.LinearFilter;
function blank(){ctx.fillStyle='#fffdf5';ctx.fillRect(0,0,512,512);texture.needsUpdate=true;}
blank();
const flyPosition=new THREE.Vector3(-1.5,-1.5,0),screenPosition=new THREE.Vector3(2.4,0,0);
const facing=Math.atan2(screenPosition.y-flyPosition.y,screenPosition.x-flyPosition.x);
const canvasRig=new THREE.Group();canvasRig.name='diffusion-screen';canvasRig.position.copy(screenPosition);canvasRig.rotation.z=facing-Math.PI/2;scene.add(canvasRig);
mesh(new THREE.BoxGeometry(2.28,.12,2.64),dark,[0,.03,1.66],canvasRig);
const screen=mesh(new THREE.PlaneGeometry(2.08,2.08),new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide}),[0,-.038,1.48],canvasRig);screen.rotation.x=Math.PI/2;
screen.name='diffusion-image';
// Separate header surface: generated image textures cannot overwrite the wordmark.
const screenHeader=mesh(new THREE.PlaneGeometry(2.08,.36),new THREE.MeshBasicMaterial({color:'#fffdf5',side:THREE.DoubleSide}),[0,-.038,2.70],canvasRig);screenHeader.rotation.x=Math.PI/2;
const wordmarkTexture=new THREE.TextureLoader().load('/assets/flycasso-wordmark.png');wordmarkTexture.colorSpace=THREE.SRGBColorSpace;
const wordmark=mesh(new THREE.PlaneGeometry(.90,.90*725/2169),new THREE.MeshBasicMaterial({map:wordmarkTexture,transparent:true,side:THREE.DoubleSide,depthWrite:false}),[0,-.039,2.70],canvasRig);
wordmark.name='screen-wordmark';wordmark.rotation.x=Math.PI/2;
for(const x of [-.72,.72])mesh(new THREE.BoxGeometry(.075,.09,.46),chrome,[x,.02,.13],canvasRig);
mesh(new THREE.BoxGeometry(2.5,.48,.09),dark,[0,0,-.12],canvasRig);
const platform=mesh(new THREE.BoxGeometry(3.7,3.25,.08),new THREE.MeshStandardMaterial({color:'#dadbd0',roughness:1}),[-1.5,-1.5,-.17]);platform.rotation.z=facing;
mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({color:'#e7e8de',roughness:1}),[0,0,-.22]);
const port=mesh(new THREE.SphereGeometry(.07,16,12),signal,[-1.17,-.025,1.48],canvasRig);
canvasRig.updateMatrixWorld(true);
let cable,pulse,birth=-Infinity,travelBirth=-Infinity,version=0,grooming,frontLegs=[],groomClock=0,groomRunning=false;
let generating=Boolean(window.flyDiffusionGenerating),lastTime=0;
const waves=[],groomMatrix=new THREE.Matrix4();
function animateFeet(time){
  if(!grooming)return;
  const delta=Math.min(.05,Math.max(0,(time-lastTime)/1000));lastTime=time;
  if(reduced.matches){groomClock=0;groomRunning=false;}
  else {
    if(generating)groomRunning=true;
    if(groomRunning)groomClock+=delta;
    if(groomClock>=grooming.duration){groomClock=generating?groomClock%grooming.duration:0;groomRunning=generating;}
  }
  // Finish the rest-to-rest bout on stop; never blend separate meshes through the body.
  const frame=Math.min(grooming.frames.length-1,Math.round(groomClock/grooming.duration*(grooming.frames.length-1)));
  frontLegs.forEach((leg,i)=>{
    leg.position.copy(grooming.frames[frame][i].position);
    leg.quaternion.copy(grooming.frames[frame][i].rotation);
  });
}

async function display({images,step}) {
  const ticket=++version;
  try {
    const decoded=await Promise.all(images.map(src=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=src;})));
    if(ticket!==version)return;
    ctx.fillStyle='#fffdf5';ctx.fillRect(0,0,512,512);ctx.imageSmoothingEnabled=false;
    const columns=Math.ceil(Math.sqrt(decoded.length)),cell=512/columns,gap=decoded.length===1?0:6;
    decoded.forEach((im,i)=>ctx.drawImage(im,(i%columns)*cell+gap,Math.floor(i/columns)*cell+gap,cell-2*gap,cell-2*gap));
    texture.needsUpdate=true;
    if(step>0){birth=performance.now();if(birth-travelBirth>=900)travelBirth=birth;}
  }catch{ /* The original PNGs remain available in the gallery if a texture cannot decode. */ }
}
window.addEventListener('diffusion-frame',e=>display(e.detail));
window.addEventListener('diffusion-reset',()=>{version++;birth=travelBirth=-Infinity;generating=true;blank();});
window.addEventListener('diffusion-end',()=>{generating=false;});
function resize(){const w=view.clientWidth,h=view.clientHeight;if(!w||!h)return;const aspect=w/h,half=Math.max(4,2.1*aspect);camera.left=-half;camera.right=half;camera.top=half/aspect;camera.bottom=-half/aspect;camera.updateProjectionMatrix();renderer.setSize(w,h,false);}
new ResizeObserver(resize).observe(view);window.addEventListener('resize',resize);
renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();$('canvas').classList.remove('brain-ready');$('empty').hidden=false;$('empty').innerHTML='<h2>3D view paused.</h2><p>Your real images are still available below. Reload to restore the fly.</p>';});

async function load(){
  const response=await fetch('/assets/fly-scene.json');if(!response.ok)throw Error('Fly body unavailable');
  const data=await response.json(),fly=new THREE.Group();fly.name='diffusion-fly';scene.add(fly);
  flyObjects(data).forEach((object,i)=>{if(data.geoms[i].type===7)fly.add(object);});
  fly.updateMatrixWorld(true);
  const head=new THREE.Box3().setFromObject(fly.getObjectByName('nmf/c_head'));
  const cap=new THREE.Group();cap.name='brain-cap';cap.position.set((head.min.x+head.max.x)/2,0,head.max.z-.1);fly.add(cap);
  const domeGeometry=new THREE.SphereGeometry(1,32,16,0,Math.PI*2,0,Math.PI/2);domeGeometry.rotateX(Math.PI/2);
  const dome=mesh(domeGeometry,new THREE.MeshPhysicalMaterial({color:'#9ce8df',transparent:true,opacity:.42,metalness:.2,roughness:.22,side:THREE.DoubleSide,depthWrite:false}),[0,0,0],cap);dome.scale.set(.31,.38,.26);
  const rim=mesh(new THREE.TorusGeometry(1,.035,8,48),chrome,[0,0,0],cap);rim.scale.set(.31,.38,.31);
  for(let i=0;i<8;i++){
    const angle=i*Math.PI/4;
    mesh(new THREE.SphereGeometry(.026,12,8),signal,[Math.cos(angle)*.25,Math.sin(angle)*.31,.145],cap);
    const points=Array.from({length:17},(_,j)=>{const t=j/16*Math.PI/2;return new THREE.Vector3(Math.cos(angle)*.31*Math.sin(t),Math.sin(angle)*.38*Math.sin(t),.265*Math.cos(t));});
    mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points),16,.007,5,false),chrome,[0,0,0],cap);
  }
  const hub=mesh(new THREE.CylinderGeometry(.085,.085,.065,24),dark,[0,0,.285],cap);hub.rotation.x=Math.PI/2;
  mesh(new THREE.SphereGeometry(.043,16,12),signal,[0,0,.332],cap);
  for(let i=0;i<2;i++){const wave=mesh(new THREE.TorusGeometry(.32,.008,6,40),new THREE.MeshBasicMaterial({color:'#53cdbb',transparent:true,opacity:0,depthWrite:false}),[0,0,.25],cap);waves.push(wave);}
  fly.position.copy(flyPosition);fly.rotation.z=facing;fly.updateMatrixWorld(true);
  const origin=cap.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0,0,.33));
  cable=new THREE.CatmullRomCurve3([origin,origin.clone().add(new THREE.Vector3(.35,0,.4)),new THREE.Vector3(.6,-.03,2.4),port.getWorldPosition(new THREE.Vector3())]);
  mesh(new THREE.TubeGeometry(cable,64,.027,8,false),dark,[0,0,0]);
  mesh(new THREE.TubeGeometry(cable,64,.012,8,false),signal,[0,-.027,.008]);
  pulse=mesh(new THREE.SphereGeometry(.052,16,12),new THREE.MeshBasicMaterial({color:'#b0fff3'}),origin.toArray());pulse.visible=false;
  $('canvas').classList.add('brain-ready');
  if(window.flyDiffusionPreview)display(window.flyDiffusionPreview);resize();
  // Baked connected-joint poses are presentation only; neither trained model is changed.
  try{const r=await fetch('/assets/grooming.json');if(!r.ok)throw Error('Grooming clip unavailable');grooming=await r.json();grooming.frames=grooming.frames.map(frame=>frame.positions.map((p,i)=>{const r=frame.rotations[i];groomMatrix.set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);return {position:new THREE.Vector3().fromArray(p),rotation:new THREE.Quaternion().setFromRotationMatrix(groomMatrix)};}));frontLegs=grooming.names.map(name=>fly.getObjectByName(name));}
  catch{$('brain-hint').textContent='Foot animation unavailable · Cap and waves are illustrative';}
}
renderer.setAnimationLoop(time=>{
  if($('diffusion-panel').hidden||document.hidden)return;
  animateFeet(time);
  const phase=(time-birth)/900,active=phase>=0&&phase<1;
  // Received steps trigger pulses; let each reach the canvas even when GPU frames arrive quickly.
  signal.emissiveIntensity=active?(reduced.matches ? .65 : .25+1.5*Math.sin(phase*Math.PI)):.2;
  waves.forEach((wave,i)=>{const t=phase-i*.2;wave.visible=active&&!reduced.matches&&t>=0;wave.scale.setScalar(1+THREE.MathUtils.clamp(t,0,1)*1.7);wave.material.opacity=Math.max(0,(1-t)*.45);});
  if(pulse){const travel=(time-travelBirth)/900;pulse.visible=travel>=0&&travel<1&&!reduced.matches;if(pulse.visible)pulse.position.copy(cable.getPoint(travel));}
  controls.update();renderer.render(scene,camera);
});
load().catch(()=>{$('empty').innerHTML='<h2>Fly view unavailable.</h2><p>You can still generate and download images below.</p>';$('brain-reset').disabled=true;});
