// Live curves from the actual JSONL logs. No smoothing or invented measurements.
let latest;
function circuitDetails(task, config) {
  if(!['image','motor'].includes(config?.task))return;
  const column=document.querySelector(`.${task==='motor'?'motor':'diffusion'}-model`);
  const text=(row,value)=>{column.querySelector(`[data-row="${row}"] p:last-child`).textContent=value;};
  const diagram=column.querySelector('.diagram-scroll');
  const calibrated=config.recipe==='calibrated-v1';
  if(diagram.dataset.circuit!==(calibrated?'calibrated':'legacy')){
    const img=document.createElement('img');img.src=`/assets/${calibrated?'calibrated':'circuit'}-${config.task}-architecture.svg`;
    img.alt=`Circuit-first ${config.task} architecture`;img.style.width='100%';diagram.replaceChildren(img);diagram.dataset.circuit=calibrated?'calibrated':'legacy';
  }
  text('samples',task==='motor'?'Free-running house · fixed cue 0.1, 0.2, 0.3':'10 classes · seed 42');
  text('loss',task==='motor'?'Muscle MSE · blue: training states · green: held-out policy states · red: held-out demonstration states · validation uses EMA':'Image MSE across recurrent denoising sequence · EMA');
  text('checkpoint',`AdamW · initial LR ${config.lr} · batch ${config.batch} · EMA 0.999 · gradient clipping 1`);
  text('execution',task==='motor'?'15 painting muscles · 24 additional circuit-driven joints · tethered body · 50 Hz commands · 10 kHz physics':`${config.steps} DDIM steps · ${config.ticks} neural updates per step · persistent state`);
  if(task==='motor') {
    const descriptions=column.querySelectorAll('[data-row="data"] dd');
    descriptions[2].textContent='320 training · 80 validation · 80 test muscle demonstrations';
    document.getElementById('motor-measurement').textContent='Only painting is supervised. Other body joints use fixed motor-pool readouts, without sensory feedback or body collisions. Non-leg axes and gains are approximations.';
  }
  if(calibrated){
    text('samples',`Curriculum: ${config.stage_name} · EMA checkpoint`);
    text('loss',task==='motor'?'Muscle imitation + physical-reward actor–critic + value loss + posterior KL; chart: muscle MSE on executed states':'Uniform x₀ MSE · 10% class dropout · generated-state exposure up to 50%');
    text('checkpoint',`AdamW · initial LR ${config.lr} · batch ${config.batch} · EMA 0.99 · gradient clipping 1`);
    if(task==='motor')column.querySelectorAll('[data-row="data"] dd')[2].textContent='Primitive motion → one sketch → up to 1,000 training sketches per category; held-out strokes for validation';
  }
  const notes=document.querySelector('.model-notes dl');
  notes.innerHTML='<dt>Graph</dt><dd>MaleCNS · 166,700 neurons · 25,582,938 directed edges</dd><dt>Circuit-first dynamics</dt><dd>Learned edge gains, neuron biases and leak rates · fixed anatomical topology and signs · persistent state</dd><dt>Interfaces</dt><dd>Fixed sensory and task-cue electrodes and output pooling · zero learned adapter parameters</dd><dt>Stopping</dt><dd>Held-out error plateau · three learning-rate reductions · minimum 1,000 optimizer steps</dd><dt>Mapping limits</dt><dd>Rate dynamics, RGB electrodes, sensory tuning and muscle-head pooling are engineered approximations.</dd>';
  if(calibrated)notes.innerHTML='<dt>Graph</dt><dd>166,700 neurons · 25,582,938 edges · 3,868,258 shared cell-type-pair gains</dd><dt>Interfaces</dt><dd>Learned local electrode calibration; fixed anatomical port locations. Image: 311 adapter parameters. Motor: 328.</dd><dt>Training</dt><dd>Persistent neural state · eight neural ticks per action or denoising step · regularization · EMA · gradient clipping</dd><dt>Curriculum gates</dt><dd>Three consecutive passing checks. Pen position/contact or generated-image quality determine progress and checkpoint selection. Forty-eight checks without improvement stop for review.</dd><dt>Mapping</dt><dd>Engineered rate dynamics, electrodes and motor pooling. Peripheral movement has no additional task reward.</dd>';
}
function plot(task, rows) {
  const canvas=document.getElementById(`${task}-curve`), rect=canvas.getBoundingClientRect();
  if(!rect.width) return;
  const scale=Math.min(devicePixelRatio,2);canvas.width=rect.width*scale;canvas.height=200*scale;
  const c=canvas.getContext('2d');c.scale(scale,scale);
  const w=rect.width,h=200,p={l:48,r:12,t:12,b:28};
  if(!rows.length) {c.fillStyle='#62544b';c.fillText('No metrics',p.l,60);return;}
  const validation=rows.filter(r=>r.validation||Number.isFinite(r.validation_mse)).map(r=>({step:r.step,loss:r.validation_mse??r.validation?.clean_image_mse??r.validation?.stroke_mse??r.validation?.joint_mse}));
  const training=rows.map(r=>({step:r.step,loss:r.joint_loss??r.unweighted_mse??r.train_mse})).filter(r=>Number.isFinite(r.loss));
  const policy=rows.filter(r=>Number.isFinite(r.policy_control_mse)).map(r=>({step:r.step,loss:r.policy_control_mse}));
  const all=[...training.map(r=>r.loss),...validation.map(r=>r.loss),...policy.map(r=>r.loss)].filter(Number.isFinite);
  if(!all.length)return;
  const min=Math.max(0,Math.min(...all)*.8),max=Math.max(...all)*1.05||1,first=rows[0].step,last=Math.max(first+1,rows.at(-1).step);
  const x=s=>p.l+(s-first)/(last-first)*(w-p.l-p.r),y=v=>h-p.b-(v-min)/(max-min)*(h-p.t-p.b);
  c.font='11px system-ui';
  for(let i=0;i<4;i++) {const v=min+(max-min)*i/3,yy=y(v);c.strokeStyle='#c7afa0';c.beginPath();c.moveTo(p.l,yy);c.lineTo(w-p.r,yy);c.stroke();c.fillStyle='#62544b';c.fillText(v.toFixed(3),0,yy+4);}
  c.strokeStyle='#355889';c.lineWidth=1.7;c.beginPath();training.forEach((r,i)=>i?c.lineTo(x(r.step),y(r.loss)):c.moveTo(x(r.step),y(r.loss)));c.stroke();
  c.fillStyle='#b22319';for(const r of validation) {c.beginPath();c.arc(x(r.step),y(r.loss),4,0,2*Math.PI);c.fill();}
  c.fillStyle='#39723c';for(const r of policy) {c.beginPath();c.arc(x(r.step),y(r.loss),3,0,2*Math.PI);c.fill();}
  c.fillStyle='#62544b';c.fillText(`Step ${first}`,p.l,h-5);const label=`Step ${last}`;c.fillText(label,w-p.r-c.measureText(label).width,h-5);
}
async function update() {
  try {
    const response=await fetch('/api/training');if(!response.ok)throw Error('Training logs unavailable');latest=await response.json();
    for(const task of ['diffusion','motor']) {
      circuitDetails(task,latest[task].config);
      const {status,metrics}=latest[task],last=metrics.at(-1),state=document.getElementById(`${task}-training-state`);
      const stale=status.updated_at&&Date.now()/1000-status.updated_at>600;
      state.textContent=status.status==='queued'?'Waiting for GPU':status.status==='needs_review'?'Quality gate stalled':status.status==='curriculum_complete'?'Curriculum complete':status.status==='paused'?'Paused':status.status==='control_quality_reached'?'Control quality reached':status.status==='validation_plateau'?'Validation plateau':stale?'No recent log update':status.status==='step_limit'?'Step limit reached':last?'Training':'Preparing';
      document.getElementById(`${task}-training-detail`).textContent=last?`${task==='motor'?(last.phase==='strokes'?'Category strokes · ':'Pen control · '):''}Step ${last.step.toLocaleString()} · batch loss ${(last.train_loss??last.train_mse).toFixed(5)} · ${last.seconds_per_step.toFixed(1)} s/step · ${last.device==='mps'?'Apple GPU (Metal)':last.device||'CPU'}.`:'No metrics';
      if(status.stage_name)document.getElementById(`${task}-training-detail`).textContent+=` Stage: ${status.stage_name}.`;
      const policy=metrics.findLast(r=>Number.isFinite(r.policy_control_mse));
      if(policy)document.getElementById(`${task}-training-detail`).textContent+=` On-policy control MSE ${policy.policy_control_mse.toFixed(5)}.`;
      const physical=metrics.findLast(r=>r.physical)?.physical??latest[task].control?.physical;
      if(task==='motor'&&physical) document.getElementById('motor-measurement').textContent=`Held-out line overlap: ${physical.map(q=>(q.stroke_f1*100).toFixed(0)+'%').join(' / ')}. Average pen error: ${(physical.reduce((s,q)=>s+q.mean_tip_error_mm,0)/physical.length).toFixed(4)} mm.`;
      const generation=metrics.findLast(r=>r.generation)?.generation;
      if(generation) document.getElementById(`${task}-training-detail`).textContent+=` Sample recognition: ${(generation.category_accuracy*100).toFixed(0)}% (${generation.examples} images).`;
      const preview=document.getElementById(`${task}-preview`);
      if(preview){preview.onload=()=>preview.hidden=false;preview.onerror=()=>preview.hidden=true;preview.src=`/${task}-preview.png?step=${last?.step||0}`;}
      plot(task,metrics);
    }
    window.dispatchEvent(new CustomEvent('training-update',{detail:latest}));
  } catch(e) {for(const task of ['diffusion','motor'])document.getElementById(`${task}-training-state`).textContent='Connection unavailable';}
}
window.addEventListener('resize',()=>{if(latest)for(const task of ['diffusion','motor'])plot(task,latest[task].metrics);});
update();setInterval(update,15000);
