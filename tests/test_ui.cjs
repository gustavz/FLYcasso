// Run with node tests/test_ui.cjs. Exercises the actual shared reader without a browser.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('web/studio.html','utf8');
const statusUpdate=fs.readFileSync('web/training.js','utf8').match(/state\.textContent=.*?;/)[0];
const paused={state:{},status:{status:'paused'},stale:true,last:{step:3400}};
vm.runInNewContext(statusUpdate,paused);
assert.equal(paused.state.textContent,'Paused','A paused job stays paused after its logs age');
const dots=[],plotContext=new Proxy({measureText:()=>({width:30})},{get:(o,k)=>k in o?o[k]:k==='arc'?()=>dots.push(o.fillStyle):()=>{}});
const charts={window:{addEventListener(){}},devicePixelRatio:1,document:{getElementById:()=>({getBoundingClientRect:()=>({width:500}),getContext:()=>plotContext})}};
vm.runInNewContext(fs.readFileSync('web/training.js','utf8').replace(/update\(\);setInterval\(update,15000\);/,''),charts);
charts.plot('motor',[{step:1,train_mse:.003,validation_mse:.08,policy_control_mse:.002}]);
assert.deepEqual(dots,['#b22319','#39723c'],'Both held-out state distributions are plotted separately');
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(new Set(ids).size,ids.length,'Duplicate element IDs');
for(const id of ['brain-view','brain-reset','image-results','sound-toggle'])assert(ids.includes(id));
assert(!ids.includes('sound-volume'),'Volume is controlled by the device');
assert(html.includes('class="sound-waves"')&&html.includes('class="sound-muted"'),'Speaker icon shows both sound states');
assert(html.includes('Cap and waves are illustrative'));
assert(html.includes("finally{window.flyDiffusionGenerating=false;window.dispatchEvent(new Event('diffusion-end'))"),'All outcomes stop grooming');
const training=html.slice(html.indexOf('<section id="training-panel"'),html.indexOf('<footer>'));
for(const id of ['model-info','motor-info','diffusion-curve','motor-curve','motor-measurement'])assert(training.includes(`id="${id}"`));
for(const name of ['diffusion','motor']){
  assert(training.includes(`id="${name}-architecture"`),'Both model architectures are shown');
  assert(training.includes(`id="${name}-preview" class="sample-batch"`),'Sample panels share dimensions');
}
assert(html.includes('aspect-ratio:6/5; object-fit:contain'),'Use the existing diffusion batch shape without stretching');
const columns=[...training.matchAll(/<article\b[\s\S]*?<\/article>/g)].map(m=>m[0]);
assert.equal(columns.length,2);
const rows=column=>[...column.matchAll(/data-row="([^"]+)"/g)].map(m=>m[1]);
assert.deepEqual(rows(columns[0]),['status','samples','loss','architecture','data','checkpoint','execution']);
assert.deepEqual(rows(columns[0]),rows(columns[1]),'Both models keep the same comparison rows');
for(const column of columns){
  assert(column.includes('<svg class="model-diagram"'),'Architecture is a diagram');
  assert(column.includes('<desc id=')&&column.includes('marker-end="url('),'Diagrams have descriptions and directed connections');
}
for(const text of ['Lower loss alone','Two models, the same fly wiring','A thought becomes','A fly, one pen','Choose a subject.','Make a painting','Draw a little'])assert(!html.includes(text),'Promotional copy stays removed');
assert(html.includes('grid-template-rows:subgrid'),'Desktop row heights follow both columns');
for(const id of ['motor-seed','motor-shuffle'])assert(ids.includes(id),'Drawing supports repeatable seeds');
for(const asset of ['flycasso-flies.png','flycasso-wordmark.png'])assert(fs.existsSync('web/assets/'+asset)&&html.includes('/assets/'+asset),'Both logo assets ship with the app');
for(const name of ['diffusion','motor','training'])assert(html.includes(`aria-controls="${name}-panel"`));
for(const [form,button,result] of [['controls','paint','image-results'],['motor-controls','start-painting','drawing-results']]){
  const markup=html.slice(html.indexOf('<form id="'+form+'"'),html.indexOf('</form>',html.indexOf('<form id="'+form+'"')));
  assert(markup.includes('id="'+result+'"'),'Results belong to their controls');
  assert(markup.indexOf('id="'+result+'"')>markup.indexOf('id="'+button+'"'),'Results follow the generate button');
}
assert(html.includes('height:100dvh; overflow:hidden'),'App fits the viewport');
assert(html.includes('main:has(#training-panel:not([hidden])) { overflow-y:auto; }'),'Only training scrolls');
assert(!html.includes('height:540px'),'Viewers have no fixed height');
assert.equal((html.match(/disabled>Save image<\/button>/g)||[]).length,2,'Both outputs use Save image');
let selected={dataset:{download:'flycasso-cat-42-2.png'},querySelector:()=>({src:'data:image/png;base64,second'})},saved;
const saveButton={},gallery={querySelector:()=>selected};
vm.runInNewContext(html.slice(html.indexOf("  $('save-image').onclick="),html.indexOf("  $('stop-diffusion').onclick=")),{
  $:id=>id==='save-image'?saveButton:gallery,
  document:{createElement:()=>({click(){saved={href:this.href,download:this.download};}})}
});
saveButton.onclick();assert.equal(saved.href,'data:image/png;base64,second');assert.equal(saved.download,'flycasso-cat-42-2.png');
saved=null;selected=null;saveButton.onclick();assert.equal(saved,null,'No download before an image is available');
const window={};
vm.runInNewContext(html.slice(html.indexOf('window.readFlyStream='),html.indexOf('  const status=')),{window,TextDecoder,Error,JSON});
function response(text){
  const bytes=new TextEncoder().encode(text);let i=0;
  return new Response(new ReadableStream({pull(c){i<bytes.length?c.enqueue(bytes.slice(i,i+=3)):c.close();}}));
}
(async()=>{
  const packets=[];
  await window.readFlyStream(response('{"type":"frame","label":"フライ"}\n{"type":"done"}'),p=>packets.push(p));
  assert.equal(packets.length,2);assert.equal(packets[0].label,'フライ');
  await assert.rejects(window.readFlyStream(response('{"type":"frame"}\n'),()=>{}),/did not finish/);
  await assert.rejects(window.readFlyStream(response('{"type":"error","error":"model failed"}\n'),()=>{}),/model failed/);
  await assert.rejects(window.readFlyStream(new Response('{"error":"busy"}',{status:429}),()=>{}),/busy/);
  await assert.rejects(window.readFlyStream(new Response('<html>Bad gateway</html>',{status:502}),()=>{}),/Server unavailable \(502\)/);
  await assert.rejects(window.readFlyStream(response('{"type":"frame"}\n{"type":"do'),()=>{}),/Connection interrupted/);
  const tailError=new Response(new ReadableStream({start(c){c.enqueue(new TextEncoder().encode('{"type":"done","classes":["frog"]}\n'));},pull(){throw Error('Connection closed after completion');}}));
  await window.readFlyStream(tailError,packet=>assert.equal(packet.classes[0],'frog'));
  console.log('UI structure and stream reader checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
