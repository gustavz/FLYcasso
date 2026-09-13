// Real Three.js scene + streamed contact-to-canvas checks, without a browser/GPU.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
(async()=>{
  const THREE=await import('../web/vendor/three.core.js'),body={THREE};
  vm.runInNewContext(fs.readFileSync('web/fly-body.js','utf8').replace(/^import .*;$/mg,'').replace('export function','function'),body);
  const data=JSON.parse(fs.readFileSync('web/assets/fly-scene.json','utf8')),elements={},inkCalls=[],surfaceDraws=[],listeners={};
  let renderLoop,scene,packets=[],sounds=[],requests=[];
  function element(id){
    if(elements[id])return elements[id];
    const ctx={drawImage(...args){surfaceDraws.push(args);},fillRect(){},beginPath(){},moveTo(...p){if(id==='painting-preview')inkCalls.push(['from',...p]);},lineTo(){},stroke(){},arc(...p){if(id==='painting-preview')inkCalls.push(['dot',...p]);},fill(){}};
    return elements[id]={width:1024,height:1024,clientWidth:900,clientHeight:560,hidden:false,value:'',textContent:'',
      prepend(){},addEventListener(){},add(option){if(!this.value)this.value=option.value;},getContext:()=>ctx};
  }
  const context={THREE:{...THREE,TextureLoader:class{load(src){return new THREE.Texture({src});}},WebGLRenderer:class{constructor(){this.domElement=element('webgl');}setPixelRatio(){}setClearColor(){}setSize(){}setAnimationLoop(fn){renderLoop=fn;}render(s){scene=s;}}},
    OrbitControls:class{constructor(){this.target=new THREE.Vector3();}update(){}addEventListener(){}},flyObjects:body.flyObjects,
    document:{createElement:()=>element('canvas-surface'),getElementById:element,querySelectorAll:()=>[],querySelector:()=>({content:'token'}),hidden:false},
    window:{flyAudio:{start:m=>sounds.push(['start',m]),stop:m=>sounds.push(['stop',m]),pencil:f=>sounds.push(['pencil',f.pen_down])},addEventListener:(type,fn)=>listeners[type]=fn,readFlyStream:async(response,receive)=>{for(const packet of packets)receive(packet);}},
    devicePixelRatio:1,matchMedia:()=>({matches:false}),ResizeObserver:class{constructor(fn){this.fn=fn;}observe(){this.fn();}},
    fetch:async (url,options)=>{if(options?.body)requests.push(JSON.parse(options.body));return {ok:true,json:async()=>url.includes('fly-scene')?data:{classes:["cat","flower","butterfly"],checkpoint_ready:true}};},
    Image:class{set src(value){this.srcValue=value;this.complete=true;this.naturalWidth=2169;queueMicrotask(()=>this.onload());}},
    Option:class{constructor(text,value){this.value=value;}},AbortController,DOMException,setTimeout,console};
  vm.runInNewContext(fs.readFileSync('web/paint.js','utf8').replace(/^import .*;$/mg,''),context);
  await new Promise(resolve=>setImmediate(resolve));renderLoop(0);
  assert.equal(element('drawing').value,'cat','Choose a category, not a reference');
  const paper=scene.getObjectByName('painted-paper');
  assert.equal(paper.material.map.image,element('canvas-surface'),'One texture includes both the logo and the painting');
  assert(surfaceDraws.some(a=>a[0]===element('painting-preview')&&a[2]===256),'Ink occupies only the lower drawable square');
  assert(surfaceDraws.some(a=>a[0].srcValue==='/assets/flycasso-wordmark.png'&&a[2]+a[4]<256),'Logo remains inside the reserved top margin');
  paper.updateMatrixWorld(true);
  for(const [u,v] of [[0,0],[1,0],[0,1],[1,1]]){
    const world=new THREE.Vector3((u-.5)*.8,v*.8-.5,0).applyMatrix4(paper.matrixWorld);
    assert(Math.abs((.4-world.y)/.8-u)<1e-6);assert(Math.abs((1.5-world.x)/.8-(1-v))<1e-6,'Texture and physical canvas axes agree');
  }
  scene.updateMatrixWorld(true);
  const block=new THREE.Box3().setFromObject(scene.getObjectByName('canvas')),top=new THREE.Box3().setFromObject(paper);
  for(const axis of ['x','y']){assert(Math.abs(block.min[axis]-top.min[axis])<1e-6);assert(Math.abs(block.max[axis]-top.max[axis])<1e-6,'Canvas top and solid support have matching edges');}
  assert.equal(scene.getObjectByName('paper-header'),undefined,'No separate header sheet');
  assert.equal(scene.getObjectByName('rf-pen'),undefined,'No pen on the resting leg');
  assert.equal(scene.getObjectByName('nmf/rf_brush').material.opacity,0);
  const pen=scene.getObjectByName('lf-pen'),tip=scene.getObjectByName('pen-contact-tip');
  assert.equal(pen.parent,scene,'Pen uses world-space contact coordinates');
  for(const name of ['gold-clip','cap-emblem','nib-slit','nib-inlay','nib-breather'])assert(pen.getObjectByName(name),'Fountain pen detail: '+name);
  const hat=scene.getObjectByName('picasso-beret'),head=scene.getObjectByName('nmf/c_head');
  assert.equal(hat.parent,head);
  scene.updateMatrixWorld(true);
  const normal=new THREE.Vector3(0,0,1).applyQuaternion(hat.getWorldQuaternion(new THREE.Quaternion()));
  assert(normal.y<-.15&&normal.z>.95,'Beret tilts sideways, rather than floating level above the head');
  const rim=scene.getObjectByName('beret-band').geometry.parameters.path.points;
  const center=hat.getWorldPosition(new THREE.Vector3()),ray=new THREE.Raycaster();
  for(const local of rim){
    const point=local.clone().applyMatrix4(hat.matrixWorld),direction=point.clone().sub(center).normalize();
    ray.set(center.clone().addScaledVector(direction,2),direction.clone().negate());
    const scalp=ray.intersectObject(head,false)[0];
    assert(scalp&&point.distanceTo(scalp.point)<.009,'Hat opening hugs the scalp around its entire circumference');
  }
  assert(scene.getObjectByName('beret-crown'),'A shaped fabric crown replaces the floating sphere');
  assert.equal(scene.children.filter(o=>o.name==='pen-strap').length,2,'Two visible straps bind pen and foot');
  assert.equal(scene.getObjectByName('nmf/lf_brush').visible,false,'Collision sphere is hidden');
  async function run(marks,stop=false,frameCount=1){
    inkCalls.length=0;const version=paper.material.map.version;
    packets=[{type:'start',training_step:500,neurons:166700,device:'mps'},
      {type:'frames',frames:Array.from({length:frameCount},(_,i)=>({...data,time:(i+1)*.01,ink:marks,pen_down:[Boolean(marks.length)],pen_tip:marks.length?marks.at(-1)[2]:[1.1,0,.3]})),progress:1,wall_seconds:.04,foot_error_mm:.02},
      {type:'done',simulation_seconds:.04,wall_seconds:.04}];
    const finished=element('motor-controls').onsubmit({preventDefault(){}});
    await new Promise(resolve=>setImmediate(resolve));if(stop)element('stop-painting').onclick();else{renderLoop(100);renderLoop(400);}await finished;
    assert(paper.material.map.version>version);
  }
  element('motor-seed').value='314159';
  await run([[0,[1.1,0,.152],[1.1,0,.152]],[0,[1.1,0,.152],[1.2,.1,.152]]]);
  assert.deepEqual(requests.at(-1),{category:'cat',seed:314159},'The chosen seed reaches inference');
  for(const id of ['drawing','motor-seed','motor-shuffle'])assert.equal(element(id).disabled,false,'Controls recover after completion');
  assert.equal(inkCalls.filter(x=>x[0]==='dot').length,2,'First touch and moving contact reach the canvas');
  assert.equal(inkCalls[0][0],'from');assert(Math.abs(inkCalls[0][1]-512)<1e-6&&Math.abs(inkCalls[0][2]-512)<1e-6);
  assert.equal(element('motor-state').textContent,'Finished · ink on paper');
  scene.updateMatrixWorld(true);const nib=tip.getWorldPosition(new THREE.Vector3());
  assert(nib.distanceTo(new THREE.Vector3(1.2,.1,.152))<1e-9,'Nib and ink end at the exact same point');
  const nibMesh=scene.getObjectByName('nib'),vertices=nibMesh.geometry.attributes.position;
  for(let i=0;i<vertices.count;i++)assert(new THREE.Vector3().fromBufferAttribute(vertices,i).applyMatrix4(nibMesh.matrixWorld).z>=.152-1e-6,'Nib never penetrates paper');
  // Every part must remain above paper, including the broad nib at the maximum allowed tilt.
  for(const direction of [[1,0,1.5],[-1,0,1.5],[0,1,1.5],[0,-1,1.5]]){
    pen.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),new THREE.Vector3(...direction).normalize());scene.updateMatrixWorld(true);
    pen.traverse(part=>{const points=part.geometry?.attributes.position;if(points)for(let i=0;i<points.count;i++)assert(new THREE.Vector3().fromBufferAttribute(points,i).applyMatrix4(part.matrixWorld).z>=.152-1e-6,part.name+' stays above paper');});
  }
  await run([[0,[1.1,0,.152],[1.2,.1,.152]]],false,20);
  assert.equal(inkCalls.filter(x=>x[0]==='dot').length,20,'Playback catches up at real simulated time and preserves every mark');
  assert(sounds.some(s=>s[0]==='pencil'),'Playback drives pencil audio');
  assert.deepEqual(sounds.at(-1),['stop','motor'],'Completing a drawing stops its soundtrack');
  await run([]);assert.equal(inkCalls.length,0,'Lifted pens cannot invent ink');
  assert.equal(element('motor-state').textContent,'Finished · no pen contact');
  await run([[0,[1.1,0,.152],[1.2,.1,.152]]],true);assert.equal(inkCalls.filter(x=>x[0]==='dot').length,1,'Stopping preserves received ink awaiting playback');assert.equal(element('motor-state').textContent,'Stopped');
  console.log('Pen attachment, paper texture orientation, streamed dots/strokes and no-contact status passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
