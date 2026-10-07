'use strict';
const $ = id => document.getElementById(id);
const fragment = new URLSearchParams(location.hash.slice(1));
let token = fragment.get('token') || sessionStorage.getItem('fly-token') || '';
if (token) sessionStorage.setItem('fly-token', token);
if (fragment.has('token')) history.replaceState(null, '', location.pathname);
let state = null, view = 'play', selected = null, promotionMove = null, flipped = false, actionBusy = false;
let yaw = .15, pitch = -.12, zoom = 1, activeGraph = null, referenceGraph = null, hover = null;
let pointer = null, points = [], graphNodes = [], graphEdges = [], frozenWatch = null;
const pointers = new Map();
let pinchDistance = 0;
const piecePaths = {
 p:'M37 49H63L59 63L69 77H31L41 63Z M36 35a14 14 0 1 0 28 0a14 14 0 1 0 -28 0',
 n:'M28 76L34 62L54 46L46 42L30 54L21 44L36 24L43 15L48 25L61 19L73 37V76Z M44 32h3v3h-3Z',
 b:'M28 76L42 58L39 50L29 41L35 27L50 12L65 27L71 41L61 50L58 58L72 76Z M54 24L43 40',
 r:'M28 76L34 43L25 38V19H37V29H44V19H56V29H63V19H75V38L66 43L72 76Z',
 q:'M30 75L36 56L24 30L40 42L50 22L60 42L76 30L64 56L70 75Z M19 27a5 5 0 1 0 10 0a5 5 0 1 0 -10 0 M45 20a5 5 0 1 0 10 0a5 5 0 1 0 -10 0 M71 27a5 5 0 1 0 10 0a5 5 0 1 0 -10 0',
 k:'M31 76L39 57L30 40L33 31H67L70 40L61 57L69 76Z M46 31V23H39V16H46V9H54V16H61V23H54V31Z'
};
const names = {p:'pawn',n:'knight',b:'bishop',r:'rook',q:'queen',k:'king'};
function showError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
async function request(path, body) {
 const response = await fetch(path, {method:body ? 'POST':'GET', headers:{Authorization:'Bearer '+token, ...(body?{'Content-Type':'application/json'}:{})}, ...(body?{body:JSON.stringify(body)}:{})});
 const result = await response.json();
 if (response.status === 401) { $('access').hidden = false; throw new Error(result.error); }
 if (!response.ok) throw new Error(result.error || 'Request failed');
 return result;
}
async function act(action, extras={}) {
 if (actionBusy) return;
 actionBusy = true;
 try { await request('/api/action', {action, ...extras}); showError(''); await refresh(); }
 catch(error) { showError(error.message); }
 finally { actionBusy = false; }
}
async function refresh() {
 const previous = state?.version.join(':');
 state = await request('/api/state');
 if (previous !== state.version.join(':')) { selected = null; if ($('promotion').open) $('promotion').close(); }
 $('connection').textContent = 'LOCAL / CONNECTED'; $('access').hidden = true;
 render();
}
async function poll() {
 try { if (!actionBusy) await refresh(); }
 catch(error) { $('connection').textContent = 'DISCONNECTED'; showError(error.message); }
 setTimeout(poll, 650);
}
$('connect').onclick = () => { token = $('access-token').value.trim(); sessionStorage.setItem('fly-token',token); refresh().catch(e=>showError(e.message)); };
function changeView(next) {
 view = next; selected = null;
 document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
 $('workspace').hidden = view === 'settings'; $('settings').hidden = view !== 'settings';
 $('play-setup').hidden = view !== 'play'; $('training').hidden = view !== 'train';
 $('play-controls').hidden = view !== 'play'; $('watch-controls').hidden = view !== 'watch';
 $('phase-label').hidden = view !== 'train';
 if (view === 'watch') act('watch');
 render();
}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>changeView(b.dataset.view));
document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>act(b.dataset.action,{generations:Number($('generations').value)}));
$('new-game').onclick=()=>{flipped=$('side').value==='black';act('new',{side:$('side').value,difficulty:Number($('difficulty').value),model:$('model').value});};
$('flip').onclick=()=>{flipped=!flipped;renderBoard();};
$('freeze').onclick=()=>{
 if(view==='watch'){frozenWatch=frozenWatch?null:activeGraph;render();}
 else act(view==='train'?'train_freeze':'toggle_freeze');
};
$('phase').onchange=()=>render();
for (const [id,field] of [['adapt','adapt'],['telemetry','telemetry'],['save-human','save_human']]) $(id).onchange=()=>act('settings',{[field]:$(id).checked});
for (const id of ['diagnostics','activations','gradients','interval']) $(id).onchange=()=>act('train_settings',{diagnostics:$('diagnostics').checked,activations:$('activations').checked,gradients:$('gradients').checked,interval:Number($('interval').value)});
$('human-train').onclick=()=>act('train_human');
$('replay-toggle').onclick=()=>act('watch',{playing:!state?.training.watch.playing});
document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>act('watch',{step:Number(b.dataset.step)}));
$('speed').onchange=()=>act('watch',{speed:Number($('speed').value)});
function boardFen() { return view==='watch'?state.training.watch.fen:view==='train'?state.training.sample_fen:state.fen; }
function squaresFromFen(fen) {
 const squares={}; let rank=7,file=0;
 for(const c of (fen || '8/8/8/8/8/8/8/8').split(' ')[0]){
  if(c==='/'){rank--;file=0;}else if(/[1-8]/.test(c))file+=Number(c);else{squares['abcdefgh'[file]+(rank+1)]=c;file++;}
 } return squares;
}
function pieceSvg(piece) {
 const white=piece===piece.toUpperCase();
 return `<svg viewBox="0 0 100 100" aria-hidden="true"><g fill="${white?'#f1f2e7':'#1a1e18'}" stroke="${white?'#272d23':'#e6e9da'}" stroke-width="2" stroke-linejoin="round"><path d="${piecePaths[piece.toLowerCase()]}"/><path d="M29 76H71L77 87H23Z"/></g></svg>`;
}
function renderBoard() {
 if(!state)return;
 const squares=squaresFromFen(boardFen());
 const targets=selected&&view==='play'?state.legal.filter(m=>m.startsWith(selected)).map(m=>m.slice(2,4)):[];
 const fragment=document.createDocumentFragment();
 for(let row=0;row<8;row++)for(let col=0;col<8;col++){
  const rank=flipped?row:7-row,file=flipped?7-col:col,square='abcdefgh'[file]+(rank+1),piece=squares[square];
  const button=document.createElement('button');
  button.className=(file+rank)%2?'light':'dark';
  button.classList.toggle('selected',square===selected);
  button.classList.toggle('legal',targets.includes(square));
  button.classList.toggle('last',view==='play'&&!!state.last_move&&(state.last_move.startsWith(square)||state.last_move.slice(2,4)===square));
  button.setAttribute('aria-label',square+(piece?' '+(piece===piece.toUpperCase()?'white ':'black ')+names[piece.toLowerCase()]:' empty'));
  if(piece)button.innerHTML=pieceSvg(piece);
  if(row===7||col===0){const coord=document.createElement('span');coord.className='coordinate';coord.textContent=(row===7?'abcdefgh'[file]:'')+(col===0?rank+1:'');button.append(coord);}
  button.onclick=()=>boardClick(square);fragment.append(button);
 }
 // Preserve keyboard focus while polling.
 const focus=document.activeElement?.closest('#board button')?.getAttribute('aria-label')?.split(' ')[0];
 $('board').replaceChildren(fragment);
 if(focus){const index=(flipped?Number(focus[1])-1:8-Number(focus[1]))*8+(flipped?7-'abcdefgh'.indexOf(focus[0]):'abcdefgh'.indexOf(focus[0]));$('board').children[index]?.focus({preventScroll:true});}
}
function boardClick(square){
 if(view!=='play'||!state.human_turn||actionBusy)return;
 const candidates=selected?state.legal.filter(m=>m.startsWith(selected+square)):[];
 if(candidates.length>1){promotionMove=selected+square;$('promotion').showModal();return;}
 if(candidates.length===1){act('move',{move:candidates[0],version:state.version});selected=null;return;}
 selected=state.legal.some(m=>m.startsWith(square))?square:null;renderBoard();
}
document.querySelectorAll('[data-promote]').forEach(b=>b.onclick=()=>{const move=promotionMove+b.dataset.promote;$('promotion').close();act('move',{move,version:state.version});});
$('promotion-cancel').onclick=()=>$('promotion').close();
let modelSignature='';
function render(){
 if(!state)return;
 const t=state.training,p=t.progress;
 if(state.error)showError(state.error);
 $('status').textContent=view==='watch'?(t.watch.record?`Replay · ply ${t.watch.ply} / ${t.watch.record.moves.length}`:'Awaiting an archived game'):view==='train'?(t.sample_step?`Captured update ${t.sample_step}`:'Awaiting an optimizer sample'):state.status;
 $('board-label').textContent=view==='train'?'01 / THE LEARNING POSITION':view==='watch'?'01 / RECORDED SELF-PLAY':'01 / YOUR NEXT MOVE';
 $('board-caption').textContent=view==='train'?(t.sample_fen?'The same replay position, before and after an update.':'No captured board yet. Training captures arrive during optimization.'):view==='watch'?'Saved games, replayed independently of training.':`Playing as ${state.human} · ${state.model}`;
 $('moves').textContent=view==='play'?state.history.map((m,i)=>i%2===0?`${i/2+1}. ${m}`:m).join('  '):'';
 renderBoard();
 const signature=state.models.join('|');
 if(signature!==modelSignature){const previous=$('model').value;modelSignature=signature;$('model').replaceChildren(new Option('Untrained / explore',''),...state.models.map(m=>new Option(m.replace('models/',''),m)));$('model').value=state.models.includes(previous)?previous:(state.models.find(m=>m.endsWith('fly_best.pt'))||'');}
 $('model-label').textContent='Strength / unrated';
 document.querySelector('[data-action=undo]').disabled=!state.can_undo;
 document.querySelector('[data-action=resign]').disabled=state.finished;
 document.querySelector('[data-action=claim_draw]').disabled=!state.can_claim;
 $('generation').textContent=p.generation??'—';$('games').textContent=p.games_played??'—';$('updates').textContent=p.training_steps??'—';
 $('train-stage').textContent=(t.busy?'':'NEXT / ')+(p.stage||'ready').toUpperCase();$('train-notice').textContent=t.notice;
 for(const action of ['train_start','train_load'])document.querySelector(`[data-action=${action}]`).disabled=t.busy;
 for(const action of ['train_pause','train_stop','train_save'])document.querySelector(`[data-action=${action}]`).disabled=!t.busy;
 $('sample-note').textContent=`Brain samples every ${t.interval} optimizer update(s). Self-play and evaluation retain the latest sample.`;
 const e=t.evaluations.at(-1);$('evaluation-note').textContent=e?`Latest evaluation: ${e.status || 'unknown'}. Absolute Elo remains unrated.`:'No completed evaluation yet. Absolute Elo remains unrated.';
 $('loss').textContent=Number.isFinite(t.metrics.at(-1)?.loss)?t.metrics.at(-1).loss.toFixed(4):'—';drawLoss(t.metrics);
 for(const [id,value] of Object.entries({adapt:state.adapt,telemetry:state.telemetry,'save-human':state.save_human,diagnostics:t.diagnostics,activations:t.activations,gradients:t.gradients}))$(id).checked=value;
 $('interval').value=String(t.interval);$('human-status').textContent=t.human_report?`Human learning: ${t.human_report.status}`:'';
 $('replay-toggle').textContent=t.watch.playing?'Pause replay':'Play replay';$('speed').value=String(t.watch.speed);
 $('freeze').textContent=(view==='train'?t.frozen:view==='watch'?!!frozenWatch:state.frozen)?'Unfreeze':'Freeze';
 const capture=view==='train'?t[$('phase').value]:view==='watch'?(frozenWatch||t.watch.brain):state.brain;
 referenceGraph=view==='train'?t[$('phase').value==='after'?'before':'after']:null;
 setGraph(capture);
 $('brain-source').textContent=capture?`${graphNodes.length} samples · ${graphEdges.length} weighted connections · ${view==='train'?`update ${t.sample_step} / ${$('phase').value}`:view==='watch'?'recorded position':state.brain_source?.current?'current position':'previous position'} · ${capture.timestamp?.slice(11,19)||'—'} UTC`:'No measured capture available yet.';
 $('brain-empty').textContent=view==='train'?(!t.diagnostics||!t.activations?'Enable training diagnostics and activations in Settings.':t.busy?`${(p.stage||'Training').toUpperCase()} is running. Brain captures arrive at sampled optimizer updates.`:'Resume training to capture a new optimizer update.'):view==='watch'?'Replay a saved game at ¼×–2× to inspect its brain.':state.telemetry?'Waiting for a measured forward pass.':'Enable brain capture in Settings.';
}
function setGraph(capture){
 activeGraph=capture;
 const unique=new Map((capture?.nodes||[]).map(n=>[n.id,n]));graphNodes=[...unique.values()];graphEdges=capture?.edges||[];
 $('brain-empty').hidden=graphNodes.length>0;drawBrain();
}
function sizeCanvas(canvas){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);const w=Math.round(rect.width*dpr),h=Math.round(rect.height*dpr);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);return [ctx,rect.width,rect.height];}
function drawBrain(){
 const [ctx,w,h]=sizeCanvas($('brain'));if(!w||!h)return;
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const scales={};for(const n of [...graphNodes,...(referenceGraph?.nodes||[])])scales[n.layer]=Math.max(scales[n.layer]||1e-8,Math.abs(n.value));
 points=graphNodes.map(n=>{let[x,y,z]=n.xyz;[x,z]=[x*cy+z*sy,-x*sy+z*cy];[y,z]=[y*cp-z*sp,y*sp+z*cp];const u=Math.min(w/4.35,h/2.7)*zoom*4/(4+z);return {...n,x:w/2+x*u,y:h*.47+y*u,z,strength:Math.abs(n.value)/scales[n.layer]};}).sort((a,b)=>b.z-a.z);
 const byId=new Map(points.map(n=>[n.id,n]));
 hover=pointer?points.filter(n=>Math.hypot(n.x-pointer[0],n.y-pointer[1])<9).sort((a,b)=>Math.hypot(a.x-pointer[0],a.y-pointer[1])-Math.hypot(b.x-pointer[0],b.y-pointer[1]))[0]:null;
 const max=Math.max(1e-10,...graphEdges.map(e=>Math.abs(e.contribution)));
 if($('synapses').checked)for(const edge of graphEdges){const a=byId.get(edge.source+'|'+edge.source_index),b=byId.get(edge.target+'|'+edge.target_index);if(!a||!b)continue;const lit=hover&&(a.id===hover.id||b.id===hover.id),strength=Math.sqrt(Math.abs(edge.contribution)/max);ctx.strokeStyle=lit?'#fafde1':`rgba(208,218,182,${.045+.62*strength})`;ctx.lineWidth=lit?1.8:.35+strength*1.15;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo((a.x+b.x)/2,(a.y+b.y)/2-8,a.x+(b.x-a.x),b.y);ctx.stroke();}
 for(const n of points){const brightness=Math.round(55+n.strength*190),r=(.7+n.strength*1.7)*Math.min(zoom,2);ctx.fillStyle=`rgb(${brightness},${brightness},${Math.round(brightness*.94)})`;ctx.strokeStyle=ctx.fillStyle;if(n.strength>.65){ctx.shadowBlur=5;ctx.shadowColor='#d9e2c177';}ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2);if(n.value<0)ctx.stroke();else ctx.fill();ctx.shadowBlur=0;if(hover?.id===n.id){ctx.strokeStyle='#fff';ctx.beginPath();ctx.arc(n.x,n.y,7,0,Math.PI*2);ctx.stroke();}}
 if(hover){const edges=graphEdges.filter(e=>e.source+'|'+e.source_index===hover.id||e.target+'|'+e.target_index===hover.id);const e=edges.sort((a,b)=>Math.abs(b.contribution)-Math.abs(a.contribution))[0];$('hover').textContent=`${hover.layer} [${hover.index}] = ${hover.value.toPrecision(5)}`+(e?` · w ${e.weight.toPrecision(3)} · input×w ${e.contribution.toPrecision(3)}`:'');}
 else $('hover').textContent='Drag to rotate · scroll or pinch to zoom · hover to inspect';
 $('zoom-label').textContent=zoom.toFixed(1)+'×';
}
function changeZoom(factor){zoom=Math.max(.6,Math.min(4,zoom*factor));drawBrain();}
$('zoom-in').onclick=()=>changeZoom(1.2);$('zoom-out').onclick=()=>changeZoom(1/1.2);
$('reset').onclick=()=>{zoom=1;yaw=.15;pitch=-.12;drawBrain();};$('synapses').onchange=drawBrain;
$('expand').onclick=()=>{$('brain-panel').classList.toggle('expanded');$('expand').textContent=$('brain-panel').classList.contains('expanded')?'↙':'↗';drawBrain();};
$('brain').addEventListener('wheel',e=>{e.preventDefault();changeZoom(Math.exp(-e.deltaY*.001));},{passive:false});
$('brain').onpointerdown=e=>{pointers.set(e.pointerId,[e.clientX,e.clientY]);$('brain').setPointerCapture(e.pointerId);pinchDistance=pointers.size===2?Math.hypot(...[...pointers.values()][0].map((v,i)=>v-[...pointers.values()][1][i])):0;};
$('brain').onpointermove=e=>{const rect=$('brain').getBoundingClientRect();pointer=[e.clientX-rect.left,e.clientY-rect.top];const previous=pointers.get(e.pointerId);if(previous){pointers.set(e.pointerId,[e.clientX,e.clientY]);if(pointers.size===2){const[a,b]=[...pointers.values()],distance=Math.hypot(a[0]-b[0],a[1]-b[1]);if(pinchDistance)zoom=Math.max(.6,Math.min(4,zoom*distance/pinchDistance));pinchDistance=distance;}else{yaw+=(e.clientX-previous[0])*.008;pitch=Math.max(-1.3,Math.min(1.3,pitch+(e.clientY-previous[1])*.008));}}drawBrain();};
for(const event of ['pointerup','pointercancel','lostpointercapture'])$('brain').addEventListener(event,e=>{pointers.delete(e.pointerId);pinchDistance=0;});
$('brain').onpointerleave=()=>{pointer=null;drawBrain();};
$('brain').onkeydown=e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-'].includes(e.key)){e.preventDefault();if(e.key==='ArrowLeft')yaw-=.1;if(e.key==='ArrowRight')yaw+=.1;if(e.key==='ArrowUp')pitch-=.1;if(e.key==='ArrowDown')pitch+=.1;if(e.key==='+'||e.key==='=')changeZoom(1.2);if(e.key==='-')changeZoom(1/1.2);pitch=Math.max(-1.3,Math.min(1.3,pitch));drawBrain();}};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('brain-panel').classList.remove('expanded');$('expand').textContent='↗';drawBrain();}});
new ResizeObserver(()=>{drawBrain();if(state)drawLoss(state.training.metrics);}).observe($('brain'));
function drawLoss(metrics){const[ctx,w,h]=sizeCanvas($('loss-chart'));const rows=metrics.filter(m=>Number.isFinite(m.loss));ctx.strokeStyle='#393d32';ctx.beginPath();ctx.moveTo(0,h-12);ctx.lineTo(w,h-12);ctx.stroke();if(!rows.length){ctx.fillStyle='#929688';ctx.font='12px Manrope';ctx.fillText('No optimizer updates captured yet.',0,45);return;}const min=Math.min(...rows.map(r=>r.loss)),max=Math.max(...rows.map(r=>r.loss));ctx.strokeStyle='#dcdfca';ctx.lineWidth=1.5;ctx.beginPath();rows.forEach((r,i)=>{const x=8+i/Math.max(1,rows.length-1)*(w-16),y=h-18-(r.loss-min)/Math.max(max-min,.01)*(h-35);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
poll();
