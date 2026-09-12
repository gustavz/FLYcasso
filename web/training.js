// Live curves from the actual JSONL logs. No smoothing or invented measurements.
let latest;
function plot(task, rows) {
  const canvas=document.getElementById(`${task}-curve`), rect=canvas.getBoundingClientRect();
  if(!rect.width) return;
  const scale=Math.min(devicePixelRatio,2);canvas.width=rect.width*scale;canvas.height=200*scale;
  const c=canvas.getContext('2d');c.scale(scale,scale);
  const w=rect.width,h=200,p={l:48,r:12,t:12,b:28};
  if(!rows.length) {c.fillStyle='#62544b';c.fillText('No metrics',p.l,60);return;}
  const validation=rows.filter(r=>r.validation).map(r=>({step:r.step,loss:r.validation.clean_image_mse??r.validation.stroke_mse??r.validation.joint_mse}));
  const training=rows.map(r=>({step:r.step,loss:r.joint_loss??r.unweighted_mse})).filter(r=>Number.isFinite(r.loss));
  const all=[...training.map(r=>r.loss),...validation.map(r=>r.loss)].filter(Number.isFinite);
  if(!all.length)return;
  const min=Math.max(0,Math.min(...all)*.8),max=Math.max(...all)*1.05||1,first=rows[0].step,last=Math.max(first+1,rows.at(-1).step);
  const x=s=>p.l+(s-first)/(last-first)*(w-p.l-p.r),y=v=>h-p.b-(v-min)/(max-min)*(h-p.t-p.b);
  c.font='11px system-ui';
  for(let i=0;i<4;i++) {const v=min+(max-min)*i/3,yy=y(v);c.strokeStyle='#c7afa0';c.beginPath();c.moveTo(p.l,yy);c.lineTo(w-p.r,yy);c.stroke();c.fillStyle='#62544b';c.fillText(v.toFixed(3),0,yy+4);}
  c.strokeStyle='#355889';c.lineWidth=1.7;c.beginPath();training.forEach((r,i)=>i?c.lineTo(x(r.step),y(r.loss)):c.moveTo(x(r.step),y(r.loss)));c.stroke();
  c.fillStyle='#b22319';for(const r of validation) {c.beginPath();c.arc(x(r.step),y(r.loss),4,0,2*Math.PI);c.fill();}
  c.fillStyle='#62544b';c.fillText(`Step ${first}`,p.l,h-5);const label=`Step ${last}`;c.fillText(label,w-p.r-c.measureText(label).width,h-5);
}
async function update() {
  try {
    const response=await fetch('/api/training');if(!response.ok)throw Error('Training logs unavailable');latest=await response.json();
    for(const task of ['diffusion','motor']) {
      const {status,metrics}=latest[task],last=metrics.at(-1),state=document.getElementById(`${task}-training-state`);
      const stale=status.updated_at&&Date.now()/1000-status.updated_at>600;
      state.textContent=status.status==='control_quality_reached'?'Control quality reached':status.status==='validation_plateau'?'Validation plateau':stale?'No recent log update':status.status==='step_limit'?'Step limit reached':last?'Training':'Preparing';
      document.getElementById(`${task}-training-detail`).textContent=last?`${task==='motor'?(last.phase==='strokes'?'Category strokes · ':'Pen control · '):''}Step ${last.step.toLocaleString()} · batch loss ${last.train_loss.toFixed(5)} · ${last.seconds_per_step.toFixed(1)} s/step · ${last.device==='mps'?'Apple GPU (Metal)':last.device||'CPU'}.`:'No metrics';
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
