// Audio lifecycle/control checks without autoplay, a browser, or speakers.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const nodes=[],contexts=[],events={},timers=new Map(),elements={},storage=new Map([['flycasso-volume','0']]);let timerId=0;
class Param{
  constructor(){this.value=0;}
  setValueAtTime(v){assert(Number.isFinite(v));this.value=v;}
  linearRampToValueAtTime(v){this.setValueAtTime(v);}
  exponentialRampToValueAtTime(v){assert(v>0);this.setValueAtTime(v);}
  setTargetAtTime(v){this.setValueAtTime(v);}
  cancelScheduledValues(){}
}
class Node{
  constructor(type){this.type=type;for(const key of ['gain','frequency','Q','threshold','ratio'])this[key]=new Param();nodes.push(this);}
  connect(to){this.connected=to;}
  disconnect(){this.connected=null;}
  start(){this.started=true;}
  stop(){this.stopped=true;this.onended?.();}
}
class AudioContext{
  constructor(){this.currentTime=1;this.sampleRate=8000;this.destination={};contexts.push(this);}
  resume(){this.resumed=true;return Promise.resolve();}
  createGain(){return new Node('gain');}
  createOscillator(){return new Node('oscillator');}
  createDynamicsCompressor(){return new Node('compressor');}
  createBiquadFilter(){return new Node('filter');}
  createBufferSource(){return new Node('buffer');}
  createBuffer(_,n){return {getChannelData:()=>new Float32Array(n)};}
}
function element(id){return elements[id]??={value:25,setAttribute(k,v){this[k]=v;}};}
const document={hidden:false,getElementById:element,addEventListener:(n,f)=>events[n]=f};
const window={AudioContext,addEventListener:(n,f)=>events[n]=f};
vm.runInNewContext(fs.readFileSync('web/sound.js','utf8'),{document,window,localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)},
  setInterval:f=>{timers.set(++timerId,f);return timerId;},clearInterval:id=>timers.delete(id),setTimeout:f=>f(),console});
assert.equal(contexts.length,0,'Loading a page does not start audio');
events['diffusion-reset']();assert.equal(contexts.length,1);assert(contexts[0].resumed);assert.equal(timers.size,1);
assert.equal(nodes[0].gain.value,1,'Old app volume preferences no longer affect system-controlled output');
assert.equal(element('sound-toggle').title,'Mute sound');
const before=nodes.length;events['diffusion-frame']({detail:{step:1}});assert(nodes.length>before,'Received steps trigger wave tones');
const limited=nodes.length;events['diffusion-frame']({detail:{step:2}});assert.equal(nodes.length,limited,'Fast frames do not flood audio');
element('sound-toggle').onclick();assert.equal(element('sound-toggle')['aria-pressed'],'false');assert.equal(storage.get('flycasso-sound'),'off');assert.equal(nodes[0].gain.value,0);
contexts[0].currentTime+=1;events['diffusion-frame']({detail:{step:3}});assert.equal(nodes.length,limited,'Muted pulses create no notes');
assert.equal(element('sound-toggle').title,'Unmute sound');
element('sound-toggle').onclick();assert.equal(nodes[0].gain.value,1);assert.equal(element('sound-toggle')['aria-pressed'],'true');
document.hidden=true;events.visibilitychange();assert.equal(nodes[0].gain.value,0);document.hidden=false;events.visibilitychange();assert.equal(nodes[0].gain.value,1);
events['diffusion-end']();assert.equal(timers.size,0);
window.flyAudio.start('motor');const noise=nodes.findLast(n=>n.type==='buffer');assert(noise.started&&noise.loop);
const filter=noise.connected,scratch=filter.connected;
window.flyAudio.pencil({pen_down:[false],pen_tip:[1,0,.3]});assert.equal(scratch.gain.value,0,'Lifted pen is silent');
window.flyAudio.pencil({pen_down:[true],pen_tip:[1,0,.152]});assert(scratch.gain.value>0,'Paper contact creates pencil texture');
window.flyAudio.pencil({pen_down:[false],pen_tip:[1,0,.3]});assert.equal(scratch.gain.value,0);
const notes=nodes.length;for(const tick of timers.values())tick();assert(nodes.length>notes,'Drawing scene plays its original arpeggio');
window.flyAudio.stop('motor');assert.equal(timers.size,0);assert(noise.stopped);
window.flyAudio.start('motor');events.pagehide();assert.equal(timers.size,0,'Leaving the page stops all sounds');
console.log('Audio gestures, pulse timing, pencil contact, music, mute state and system-volume output checks passed.');
