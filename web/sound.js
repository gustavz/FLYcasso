// Original, synthesized scene audio. No recordings, downloads or music service.
(()=>{
  const button=document.getElementById('sound-toggle');
  let context,master,enabled=true;
  const scenes={};
  try{enabled=localStorage.getItem('flycasso-sound')!=='off';}catch{}
  function level(){
    button.setAttribute('aria-pressed',String(enabled));button.setAttribute('aria-label',button.disabled?'Sound unavailable':'Sound');
    button.title=button.disabled?'Sound unavailable':enabled?'Mute sound':'Unmute sound';
    if(master){master.gain.cancelScheduledValues(context.currentTime);master.gain.setTargetAtTime(enabled&&!document.hidden?1:0,context.currentTime,.04);}
  }
  function audio(){
    if(!context){
      const Audio=window.AudioContext||window.webkitAudioContext;
      if(!Audio){enabled=false;button.disabled=true;level();return false;}
      context=new Audio();master=context.createGain();master.gain.value=0;
      const limiter=context.createDynamicsCompressor();limiter.threshold.value=-18;limiter.ratio.value=4;
      master.connect(limiter);limiter.connect(context.destination);level();
    }
    if(enabled)context.resume().catch(()=>{});
    return true;
  }
  function source(scene,node){
    scene.sources.add(node);node.onended=()=>{scene.sources.delete(node);node.disconnect();};return node;
  }
  function note(scene,hz,length=.9,loudness=.06,slide=hz){
    if(!enabled||document.hidden)return;
    const now=context.currentTime;
    for(const [harmonic,amplitude] of [[1,1],[2,.18]]){
      const oscillator=source(scene,context.createOscillator()),gain=context.createGain();
      oscillator.type='sine';oscillator.frequency.setValueAtTime(hz*harmonic,now);oscillator.frequency.exponentialRampToValueAtTime(slide*harmonic,now+length);
      gain.gain.setValueAtTime(0,now);gain.gain.linearRampToValueAtTime(loudness*amplitude,now+.012);gain.gain.exponentialRampToValueAtTime(.0001,now+length);
      oscillator.connect(gain);gain.connect(scene.gain);const ended=oscillator.onended;
      oscillator.onended=()=>{ended();gain.disconnect();};oscillator.start(now);oscillator.stop(now+length+.02);
    }
  }
  function stop(mode){
    const scene=scenes[mode];if(!scene)return;
    delete scenes[mode];clearInterval(scene.timer);
    scene.gain.gain.setTargetAtTime(0,context.currentTime,.06);
    for(const node of scene.sources)try{node.stop(context.currentTime+.3);}catch{}
    setTimeout(()=>scene.gain.disconnect(),400);
  }
  function start(mode){
    stop(mode);if(!audio())return;
    const scene={gain:context.createGain(),sources:new Set(),beat:0,lastPulse:-Infinity,pen:false,lastTip:null,lastContact:-Infinity};
    scene.gain.gain.value=0;scene.gain.gain.setTargetAtTime(1,context.currentTime,.15);scene.gain.connect(master);scenes[mode]=scene;
    if(mode==='diffusion'){
      const filter=context.createBiquadFilter();filter.type='lowpass';filter.frequency.value=500;filter.Q.value=.8;filter.connect(scene.gain);
      for(const hz of [55,110,110.7]){
        const oscillator=source(scene,context.createOscillator()),gain=context.createGain();oscillator.type='sine';oscillator.frequency.value=hz;gain.gain.value=.028;
        oscillator.connect(gain);gain.connect(filter);oscillator.start();
        const ended=oscillator.onended;oscillator.onended=()=>{ended();gain.disconnect();};
      }
      scene.timer=setInterval(()=>{if(!document.hidden&&enabled)filter.frequency.setTargetAtTime(380+160*Math.sin(context.currentTime*.7),context.currentTime,.2);},120);
    }else{
      const buffer=context.createBuffer(1,context.sampleRate*2,context.sampleRate),data=buffer.getChannelData(0);
      for(let i=0;i<data.length;i++)data[i]=Math.random()*2-1;
      const noise=source(scene,context.createBufferSource());noise.buffer=buffer;noise.loop=true;
      scene.filter=context.createBiquadFilter();scene.filter.type='bandpass';scene.filter.frequency.value=1900;scene.filter.Q.value=.65;
      scene.scratch=context.createGain();scene.scratch.gain.value=0;
      noise.connect(scene.filter);scene.filter.connect(scene.scratch);scene.scratch.connect(scene.gain);noise.start();
      const chords=[[220,261.63,329.63],[174.61,220,261.63],[196,261.63,329.63],[196,246.94,293.66]],pattern=[0,1,2,1,2,1];
      scene.timer=setInterval(()=>{
        if(context.currentTime-scene.lastContact>.15)scene.scratch.gain.setTargetAtTime(0,context.currentTime,.03);
        if(!enabled||document.hidden)return;
        const beat=scene.beat++,chord=chords[Math.floor(beat/6)%chords.length];
        if(beat%6===0)note(scene,chord[0]/2,1.7,.065);
        note(scene,chord[pattern[beat%6]],1.05,beat%3===0?.055:.036);
      },330);
    }
  }
  function pencil(frame){
    const scene=scenes.motor;if(!scene)return;
    const down=Boolean(frame.pen_down?.[0]),point=frame.pen_tip;
    const distance=point&&scene.lastTip?Math.hypot(...point.map((v,i)=>v-scene.lastTip[i])):0;
    const loudness=down?Math.min(.11,.025+distance*18):0;
    scene.scratch.gain.cancelScheduledValues(context.currentTime);scene.scratch.gain.setTargetAtTime(loudness,context.currentTime,.015);
    scene.filter.frequency.setTargetAtTime(1400+Math.min(1800,distance*200000),context.currentTime,.025);
    if(down&&!scene.pen)note(scene,740,.045,.012);
    scene.pen=down;scene.lastTip=point;scene.lastContact=context.currentTime;
  }
  button.onclick=()=>{enabled=!enabled;if(!audio())return;level();try{localStorage.setItem('flycasso-sound',enabled?'on':'off');}catch{}};
  document.addEventListener('visibilitychange',level);
  window.addEventListener('pagehide',()=>Object.keys(scenes).forEach(stop));
  window.addEventListener('diffusion-reset',()=>start('diffusion'));
  window.addEventListener('diffusion-end',()=>stop('diffusion'));
  window.addEventListener('diffusion-frame',({detail})=>{
    const scene=scenes.diffusion;if(!scene||!detail.step||context.currentTime-scene.lastPulse<.25)return;
    scene.lastPulse=context.currentTime;const hz=[220,277.18,329.63,440,554.37][detail.step%5];note(scene,hz,.65,.05,hz*1.5);
  });
  window.flyAudio={start,stop,pencil};level();
})();
