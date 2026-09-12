import * as THREE from 'three';
import {OrbitControls} from '/assets/OrbitControls.js';
import {flyObjects} from '/fly-body.js';

const $=id=>document.getElementById(id),view=$('fly-view'),status=$('motor-status');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#eee8e2');view.prepend(renderer.domElement);
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(40,1,.01,50);
camera.up.set(0,0,1);camera.position.set(3.3,-3.3,3.0);
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(.3,0,.75);controls.enableDamping=!matchMedia('(prefers-reduced-motion: reduce)').matches;controls.minDistance=.6;controls.maxDistance=8;controls.update();
// Keep the same Z-up axis for OrbitControls; a tiny overhead offset avoids a pole singularity.
const cameras={orbit:[[3.3,-3.3,3],[.3,0,.75]],paper:[[1.0999,0,2.8],[1.1,0,.15]],side:[[.6,-4,1.1],[.5,0,.65]]};
const cameraButtons=[...document.querySelectorAll('[data-camera]')];
cameraButtons.forEach(button=>button.onclick=()=>{
  const [position,target]=cameras[button.dataset.camera];
  // Drain any previous orbit inertia before jumping to a preset.
  const damping=controls.enableDamping;controls.enableDamping=false;controls.update();
  camera.position.set(...position);controls.target.set(...target);controls.update();controls.enableDamping=damping;
  cameraButtons.forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
});
controls.addEventListener('start',()=>cameraButtons.forEach(b=>b.setAttribute('aria-pressed','false')));
scene.add(new THREE.HemisphereLight(0xffffff,0x4a4c38,2.5));
const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(1,-2,5);scene.add(light);
let objects=[],queue=[],transition=null,playbackAt=null,initialPose,aborter=null,categories=[],ready=false,motorStep=0,markCount=0;
const inkColors=['#86508d'],painting=$('painting-preview');
const pc=painting.getContext('2d'),surface=document.createElement('canvas');
surface.width=1024;surface.height=1280;
const surfaceContext=surface.getContext('2d'),paperTexture=new THREE.CanvasTexture(surface);
paperTexture.colorSpace=THREE.SRGBColorSpace;
// One continuous canvas; the upper 20% is a logo margin outside the contact area.
const paper=new THREE.Mesh(new THREE.PlaneGeometry(.8,1),new THREE.MeshBasicMaterial({map:paperTexture,side:THREE.DoubleSide}));
paper.name='painted-paper';paper.position.set(1.2,0,.152);paper.rotation.z=-Math.PI/2;scene.add(paper);
const wordmark=new Image();wordmark.onload=updatePaper;wordmark.src='/assets/flycasso-wordmark.png';
function updatePaper() {
  surfaceContext.fillStyle='#fffdf5';surfaceContext.fillRect(0,0,1024,256);
  surfaceContext.drawImage(painting,0,256,1024,1024);
  if(wordmark.complete&&wordmark.naturalWidth)surfaceContext.drawImage(wordmark,294,55,436,436*725/2169);
  paperTexture.needsUpdate=true;
}

let pen,penTip,penStraps=[];
const penAxis=new THREE.Vector3(),penUp=new THREE.Vector3(0,0,1);
function attachPens() {
  const brush=objects.find(o=>o.name==='nmf/lf_brush');brush.visible=false;
  pen=new THREE.Group();pen.name='lf-pen';scene.add(pen);
  const resin=new THREE.MeshPhysicalMaterial({color:'#101014',roughness:.19,metalness:.05,clearcoat:1,clearcoatRoughness:.12});
  const gold=new THREE.MeshStandardMaterial({color:'#d5ad57',metalness:.65,roughness:.24});
  const platinum=new THREE.MeshStandardMaterial({color:'#dbdce0',metalness:.6,roughness:.22});
  const black=new THREE.MeshStandardMaterial({color:'#141316',roughness:.65});
  // Posted cap: the open nib writes at local origin; the cap sits on the rear of the barrel.
  for(const [name,profile,material] of [
    ['barrel',[[0,.074],[.010,.074],[.017,.083],[.018,.11],[.019,.145],[.023,.16],[.025,.22],[.024,.29],[.019,.36],[0,.37]],resin],
    ['cap',[[0,.292],[.027,.292],[.030,.299],[.031,.33],[.030,.414],[.027,.448],[.020,.468],[.010,.475],[0,.477]],resin]]) {
    const part=new THREE.Mesh(new THREE.LatheGeometry(profile.map(p=>new THREE.Vector2(...p)),40),material);
    part.name=name;part.rotation.x=Math.PI/2;pen.add(part);
  }
  for(const [z,radius,width] of [[.146,.020,.004],[.296,.028,.003],[.306,.031,.009],[.319,.031,.003],[.449,.027,.004]]) {
    const ring=new THREE.Mesh(new THREE.CylinderGeometry(radius,radius,width,40),gold);
    ring.name='gold-band';ring.rotation.x=Math.PI/2;ring.position.z=z;pen.add(ring);
  }
  // Curved, broad-shouldered nib with a silver centre, slit and breather hole.
  const nibVertices=[],nibIndices=[],nibProfile=[[0,0,0],[.008,.003,.001],[.032,.017,.004],[.058,.014,.005],[.083,.010,.003]];
  nibProfile.forEach(([z,width,curve],row)=>{
    for(const u of [-1,0,1])nibVertices.push(u*width,-curve*(1-u*u*.65),z);
    if(row)for(let col=0;col<2;col++){const a=(row-1)*3+col,b=a+3;nibIndices.push(a,b,a+1,a+1,b,b+1);}
  });
  const nibGeometry=new THREE.BufferGeometry();nibGeometry.setAttribute('position',new THREE.Float32BufferAttribute(nibVertices,3));nibGeometry.setIndex(nibIndices);nibGeometry.computeVertexNormals();
  const nib=new THREE.Mesh(nibGeometry,gold.clone());nib.material.side=THREE.DoubleSide;nib.name='nib';pen.add(nib);
  const inlay=new THREE.Mesh(nibGeometry,platinum);inlay.material.side=THREE.DoubleSide;inlay.scale.x=.48;inlay.scale.y=1.05;inlay.name='nib-inlay';pen.add(inlay);
  const slit=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,-.0007,.003),new THREE.Vector3(0,-.0046,.032),new THREE.Vector3(0,-.0056,.055)]),new THREE.LineBasicMaterial({color:'#161412'}));
  slit.name='nib-slit';pen.add(slit);
  const hole=new THREE.Mesh(new THREE.CircleGeometry(.0023,16),black);hole.rotation.x=Math.PI/2;hole.position.set(0,-.0057,.055);hole.name='nib-breather';pen.add(hole);
  const clipPath=new THREE.CatmullRomCurve3([[0,-.024,.449],[0,-.038,.439],[0,-.038,.387],[0,-.037,.346],[0,-.032,.34]].map(p=>new THREE.Vector3(...p)));
  const clip=new THREE.Mesh(new THREE.TubeGeometry(clipPath,24,.0028,8,false),gold);clip.name='gold-clip';pen.add(clip);
  const emblemShape=new THREE.Shape();
  for(let i=0;i<=96;i++){const angle=i/96*Math.PI*2,radius=.0095+.0025*Math.cos(6*angle),x=radius*Math.cos(angle),y=radius*Math.sin(angle);if(i)emblemShape.lineTo(x,y);else emblemShape.moveTo(x,y);}
  const emblem=new THREE.Mesh(new THREE.ShapeGeometry(emblemShape),new THREE.MeshStandardMaterial({color:'#fffdf4',roughness:.3,side:THREE.DoubleSide}));
  emblem.name='cap-emblem';emblem.position.z=.4775;pen.add(emblem);
  penTip=new THREE.Object3D();penTip.name='pen-contact-tip';pen.add(penTip);
  for(const name of ['nmf/lf_tarsus4','nmf/lf_tarsus5']) {
    const strap=new THREE.Mesh(new THREE.TorusGeometry(1,.10,8,32),new THREE.MeshStandardMaterial({color:'#e5bc76',roughness:.9}));
    strap.name='pen-strap';scene.add(strap);penStraps.push({strap,foot:objects.find(o=>o.name===name)});
  }
  const head=objects.find(o=>o.name==='nmf/c_head');head.updateMatrixWorld(true);
  head.geometry.computeBoundingBox();
  const bounds=head.geometry.boundingBox.clone().applyMatrix4(head.matrixWorld);
  const height=bounds.max.z-bounds.min.z,hat=new THREE.Group();hat.name='picasso-beret';
  hat.position.set((bounds.min.x+bounds.max.x)/2,0,bounds.max.z-height*.20);
  hat.rotation.set(.20,-.045,0);hat.updateMatrixWorld(true);
  const felt=new THREE.MeshStandardMaterial({color:'#252429',roughness:1});
  // Fit the tilted opening to the actual scalp, excluding eyes and decorative hairs.
  const rim=[],ray=new THREE.Raycaster(),center=hat.position.clone();
  for(let i=0;i<48;i++) {
    const angle=i/48*Math.PI*2,direction=new THREE.Vector3(Math.cos(angle),Math.sin(angle),0).applyQuaternion(hat.quaternion);
    ray.set(center.clone().addScaledVector(direction,2),direction.clone().negate());
    const hit=ray.intersectObject(head,false)[0];
    const point=hat.worldToLocal(hit.point.clone());point.addScaledVector(new THREE.Vector3(Math.cos(angle),Math.sin(angle),0),.006);rim.push(point);
  }
  const band=new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rim,true),96,.011,8,true),felt);
  band.name='beret-band';hat.add(band);
  const vertices=[],indices=[],profile=[[1,0],[1.08,.025],[1.18,.055],[1.15,.09],[.92,.135],[.55,.165],[0,.18]];
  profile.forEach(([radius,z],row)=>rim.forEach((point,i)=>{
    const drape=Math.sin(Math.PI*row/(profile.length-1));
    const x=point.x*radius-.012*drape,y=point.y*radius-.035*drape;
    ray.set(new THREE.Vector3(x,y,1).applyMatrix4(hat.matrixWorld),new THREE.Vector3(0,0,-1).applyQuaternion(hat.quaternion));
    const scalp=ray.intersectObject(head,false)[0];
    const clearance=scalp?hat.worldToLocal(scalp.point.clone()).z+.018:0;
    vertices.push(x,y,row?Math.max(z,clearance):0);
    if(row<profile.length-1){const a=row*48+i,b=row*48+(i+1)%48;indices.push(a,b,a+48,b,b+48,a+48);}
  }));
  const shape=new THREE.BufferGeometry();shape.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));shape.setIndex(indices);shape.computeVertexNormals();
  const crown=new THREE.Mesh(shape,felt);crown.name='beret-crown';hat.add(crown);
  const stem=new THREE.Mesh(new THREE.CylinderGeometry(.009,.012,.035,10),felt);stem.rotation.x=Math.PI/2;stem.position.z=.195;hat.add(stem);
  scene.add(hat);head.attach(hat);
  const hairs=head.getObjectByName('bristles'),hairMatrix=new THREE.Matrix4();
  if(hairs)for(let i=0;i<hairs.count;i++){
    hairs.getMatrixAt(i,hairMatrix);
    const center=new THREE.Vector3().setFromMatrixPosition(hairMatrix).applyMatrix4(hairs.matrixWorld);
    if(hat.worldToLocal(center).z>-.015)hairs.setMatrixAt(i,new THREE.Matrix4().makeScale(0,0,0));
  }
  if(hairs)hairs.instanceMatrix.needsUpdate=true;
}
function updatePen(point) {
  if(!pen)return;
  const brush=objects.find(o=>o.name==='nmf/lf_brush');
  pen.position.copy(point??brush.position.clone().add(new THREE.Vector3(0,0,-.025)));
  pen.position.z=Math.max(paper.position.z,pen.position.z);
  // A raised barrel sits beside the tarsus; its nib is the recorded contact point.
  penAxis.copy(objects.find(o=>o.name==='nmf/lf_tarsus3').position).sub(pen.position);
  penAxis.z=Math.max(.16,penAxis.z,1.5*Math.hypot(penAxis.x,penAxis.y));penAxis.normalize();pen.quaternion.setFromUnitVectors(penUp,penAxis);
  for(const {strap,foot} of penStraps) {
    const along=THREE.MathUtils.clamp(foot.position.clone().sub(pen.position).dot(penAxis),.08,.22);
    const barrel=pen.position.clone().addScaledVector(penAxis,along);
    const across=barrel.clone().sub(foot.position),distance=across.length();
    if(distance<1e-8)across.set(1,0,0);else across.normalize();const normal=new THREE.Vector3().crossVectors(across,penAxis).normalize();
    const side=new THREE.Vector3().crossVectors(normal,across).normalize();
    strap.position.copy(barrel).add(foot.position).multiplyScalar(.5);
    strap.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(across,side,normal));
    strap.scale.set(distance/2+.033,.034,.034);
  }
}
const rotationMatrix=new THREE.Matrix4();
function pose(frame) {
  objects.forEach((o,i)=>{
    o.position.fromArray(frame.positions[i]);const r=frame.rotations[i];
    rotationMatrix.set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);
    o.quaternion.setFromRotationMatrix(rotationMatrix);
  });
  updatePen(frame.pen_tip?new THREE.Vector3().fromArray(frame.pen_tip):undefined);
  addInk(frame.ink||[]);
}
function addInk(marks) {
  const toPixel=p=>[(.4-p[1])/.8*painting.width,(1.5-p[0])/.8*painting.height];
  for(const [side,a,b] of marks) {
    const from=toPixel(a),to=toPixel(b);pc.strokeStyle=pc.fillStyle=inkColors[side];pc.lineWidth=8;pc.lineCap='round';
    pc.beginPath();pc.moveTo(...from);pc.lineTo(...to);pc.stroke();
    // Canvas implementations may discard a zero-length line; stamp the contact explicitly.
    pc.beginPath();pc.arc(...to,4,0,Math.PI*2);pc.fill();markCount++;
  }
  if(marks.length)updatePaper();
}
function clearInk() {markCount=0;pc.fillStyle='#fffdf5';pc.fillRect(0,0,painting.width,painting.height);updatePaper();}
function resize() {const w=view.clientWidth,h=view.clientHeight;if(!w)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h,false);}
window.addEventListener('resize',resize);new ResizeObserver(resize).observe(view);
renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();aborter?.abort();status.textContent='3D graphics paused. Reload this tab to restore the view.';});
renderer.setAnimationLoop(time=>{
  if(playbackAt===null&&queue.length)playbackAt=time;
  while(transition||queue.length) {
  if(!transition) {
    const frame=queue.shift();
    const destinations=objects.map((o,i)=>{const r=frame.rotations[i];rotationMatrix.set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);return {from:o.position.clone(),to:new THREE.Vector3().fromArray(frame.positions[i]),q0:o.quaternion.clone(),q1:new THREE.Quaternion().setFromRotationMatrix(rotationMatrix)};});
    transition={frame,destinations,start:playbackAt,duration:10,tip0:pen.position.clone(),tip1:frame.pen_tip?new THREE.Vector3().fromArray(frame.pen_tip):new THREE.Vector3().fromArray(frame.positions[objects.findIndex(o=>o.name==='nmf/lf_brush')]).add(new THREE.Vector3(0,0,-.025))};
  }
  if(transition) {
    const alpha=Math.min(1,(time-transition.start)/transition.duration);
    objects.forEach((o,i)=>{const d=transition.destinations[i];o.position.lerpVectors(d.from,d.to,alpha);o.quaternion.slerpQuaternions(d.q0,d.q1,alpha);});
    updatePen(new THREE.Vector3().lerpVectors(transition.tip0,transition.tip1,alpha));
    if(alpha===1){addInk(transition.frame.ink||[]);window.flyAudio?.pencil(transition.frame);if(aborter)$('motor-state').textContent=transition.frame.pen_down?.some(Boolean)?'Pen touching paper':'Moving pen above paper';playbackAt=transition.start+transition.duration;transition=null;}
    else break;
  }
  }
  if(!transition&&!queue.length)playbackAt=null;
  controls.update();if(!$('motor-panel').hidden&&!document.hidden)renderer.render(scene,camera);
});

async function load() {
  const [a,b]=await Promise.all([fetch('/assets/fly-scene.json'),fetch('/api/paint-info')]);
  if(!a.ok||!b.ok)throw Error('Prepare the body assets and stroke dataset first.');
  const data=await a.json(),info=await b.json();
  objects=flyObjects(data);
  objects.find(o=>o.name==='canvas').geometry.scale(1.25,1,1).translate(.1,0,0);
  attachPens();scene.add(...objects);
  initialPose=data;pose(data);clearInk();resize();
  categories=info.classes;categories.forEach(name=>$('drawing').add(new Option(name[0].toUpperCase()+name.slice(1),name)));
  $('drawing').disabled=false;ready=info.checkpoint_ready;
  $('start-painting').disabled=!ready;$('start-painting').textContent=ready?'Draw sketch':'Waiting for trained weights…';
  $('motor-state').textContent=ready?'Ready':'Checkpoint unavailable';status.textContent='';
}
window.addEventListener('training-update',async()=>{
  if(ready||aborter)return;
  try{const response=await fetch('/api/paint-info');if(!response.ok)return;const info=await response.json();ready=info.checkpoint_ready;
    if(JSON.stringify(categories)!==JSON.stringify(info.classes)){categories=info.classes;$('drawing').innerHTML='';categories.forEach(name=>$('drawing').add(new Option(name[0].toUpperCase()+name.slice(1),name)));}
    if(ready){$('start-painting').disabled=false;$('start-painting').textContent='Draw sketch';}}catch{}
});
$('motor-controls').onsubmit=async e=>{
  e.preventDefault();if(!ready)return;window.flyAudio?.start('motor');aborter=new AbortController();$('start-painting').disabled=true;for(const id of ['drawing','motor-seed','motor-shuffle'])$(id).disabled=true;$('stop-painting').hidden=false;$('save-painting').disabled=true;clearInk();queue=[];transition=null;playbackAt=null;pose(initialPose);
  $('motor-progress').value=0;$('motor-state').textContent='Starting…';$('motor-measurement').textContent='No execution measurements';status.textContent='Initializing simulation…';
  try {
    const response=await fetch('/api/paint',{method:'POST',signal:aborter.signal,headers:{'Content-Type':'application/json','X-Flycasso-Token':document.querySelector('meta[name=flycasso-token]').content},body:JSON.stringify({category:$('drawing').value,seed:Number($('motor-seed').value)})});
    await window.readFlyStream(response,packet=>{
      if(packet.type==='planning'){$('motor-state').textContent='Sampling strokes…';status.textContent='Stroke diffusion running';}
      if(packet.type==='start'){motorStep=packet.training_step;$('motor-info').textContent=`Motor checkpoint: step ${packet.training_step.toLocaleString()} · ${packet.neurons.toLocaleString()} neurons · ${packet.device==='mps'?'Apple GPU (Metal)':packet.device||'CPU'} · separate weights`;}
      if(packet.type==='frames'){
        // Bound playback lag after returning from a backgrounded phone tab; preserve every ink mark.
        if(queue.length>120){for(const frame of queue)addInk(frame.ink||[]);queue=[];}
        queue.push(...packet.frames);const sim=packet.frames.at(-1).time;
        $('motor-progress').value=packet.progress;
        status.textContent=`Live simulation · ${sim.toFixed(1)} simulated seconds · ${(sim/packet.wall_seconds).toFixed(2)}× real-time speed`;
        $('motor-measurement').textContent=`Foot-to-target error: ${packet.foot_error_mm.toFixed(3)} mm (latest action).`;
      }
      if(packet.type==='done'){$('motor-state').textContent='Finishing playback…';status.textContent=`${packet.simulation_seconds.toFixed(1)} simulated seconds in ${packet.wall_seconds.toFixed(1)} seconds.`;}
    });
    while(queue.length||transition){if(aborter.signal.aborted)throw new DOMException('Stopped','AbortError');await new Promise(resolve=>setTimeout(resolve,50));}
    $('motor-state').textContent=markCount?'Finished · ink on paper':'Finished · no pen contact';if(!markCount)status.textContent='This checkpoint kept the pen above the paper. Try again after more training.';
  }catch(error){if(transition)addInk(transition.frame.ink||[]);for(const frame of queue)addInk(frame.ink||[]);$('motor-state').textContent=error.name==='AbortError'?'Stopped':'Interrupted';status.textContent=error.name==='AbortError'?'Painting stopped.':error.message;queue=[];transition=null;}
  finally{window.flyAudio?.stop('motor');aborter=null;$('start-painting').disabled=false;for(const id of ['drawing','motor-seed','motor-shuffle'])$(id).disabled=false;$('stop-painting').hidden=true;$('save-painting').disabled=false;}
};
$('stop-painting').onclick=()=>aborter?.abort();
$('save-painting').onclick=()=>{const link=document.createElement('a');link.download=`flycasso-front-legs-step${motorStep}-${$('drawing').value}.png`;link.href=painting.toDataURL('image/png');link.click();};
load().catch(error=>{$('motor-state').textContent='Unavailable';status.textContent=error.message;$('start-painting').textContent='Motor studio unavailable';});
