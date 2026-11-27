/* Camera viewfinder for the design preview.
   Synthetic wall, gyms and recognition results only: nothing here reads a camera,
   location or the gym pack. app.js calls window.BBCapture(api) once. */
window.BBCapture=function(api){
'use strict';
const {state,go,render,toast,gradeOptions,getRecords,addRecord}=api;

const GYMS=[
  {name:'示例岩馆',km:.8,kinds:['抱石','绳索']},
  {name:'示例抱石馆',km:2.4,kinds:['抱石']},
  {name:'示例攀岩中心',km:2.9,kinds:['抱石','绳索']},
  {name:'示例岩壁 · 东站',km:5.1,kinds:['绳索']},
  {name:'示例抱石工作室',km:7.6,kinds:['抱石']},
  {name:'示例攀岩空间',km:11.2,kinds:['抱石','绳索']},
  {name:'示例户外岩场',km:38,kinds:['绳索']}
];
const DISCIPLINES=['抱石','顶绳','先锋'];
const REASON={'顶绳':'顶端有固定保护点，判断为顶绳','先锋':'有快挂，绳从下往上，判断为先锋','抱石':'没有绳，地上有垫，判断为抱石'};
const ACCENTS={'攀岩橙':'#ff7a3d','岩点绿':'#3fc89a','镁粉白':'#f2efe8'};

const cam={
  scene:'顶绳',          // what the simulated wall shows
  mode:'auto',           // 'auto' | 'manual'
  discipline:null,       // null while the first scan runs
  scanning:false,
  grade:{'抱石':'V4','绳索':'5.11a'},
  gym:null,gymNote:'',custom:[],
  filter:'全部',query:'',
  sound:true,accent:'攀岩橙',
  front:false,menu:false,sheet:false,
  startedAt:0,timers:[],wheels:[]
};
const scale=d=>d==='抱石'?'抱石':'绳索';
const gradeOf=()=>cam.grade[scale(cam.discipline||cam.scene)];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const km=g=>g.km==null?'自定义':`约 ${g.km<10?g.km.toFixed(1):Math.round(g.km)} km`;
const later=(fn,ms)=>{const t=setTimeout(fn,ms);cam.timers.push(t);return t;};
const $=s=>document.querySelector('#native-screen '+s);

/* ---------- Sound and touch ---------- */
let actx=null;
function tick(pitch=1){
  if(cam.sound){try{
    actx=actx||new (window.AudioContext||window.webkitAudioContext)();
    if(actx.state==='suspended')actx.resume();
    const t=actx.currentTime,len=Math.floor(actx.sampleRate*.014),buf=actx.createBuffer(1,len,actx.sampleRate),d=buf.getChannelData(0);
    for(let i=0;i<len;i++)d[i]=(Math.random()*2-1)*Math.pow(1-i/len,2);
    const noise=actx.createBufferSource();noise.buffer=buf;
    const band=actx.createBiquadFilter();band.type='bandpass';band.frequency.value=3400*pitch;band.Q.value=5;
    const ng=actx.createGain();ng.gain.value=.5;
    noise.connect(band).connect(ng).connect(actx.destination);noise.start(t);
    const osc=actx.createOscillator(),og=actx.createGain();osc.type='sine';
    osc.frequency.setValueAtTime(1400*pitch,t);osc.frequency.exponentialRampToValueAtTime(420*pitch,t+.04);
    og.gain.setValueAtTime(.09,t);og.gain.exponentialRampToValueAtTime(.0001,t+.05);
    osc.connect(og).connect(actx.destination);osc.start(t);osc.stop(t+.06);
  }catch{}}
  try{navigator.vibrate?.(6);}catch{}
}

/* ---------- A drum / wheel shared by grade and gym pickers ---------- */
function wheel(el,o){
  // o: {items:[{label,sub}], index, axis:'x'|'y', step(deg), radius(px), pitch, onSettle(i)}
  let pos=o.index,last=Math.round(pos),anim=0,nodes=[];
  const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
  const pxPer=()=>o.radius*o.step*Math.PI/180;
  function build(){
    el.innerHTML=o.items.map((x,i)=>`<button class="wheel-item" data-i="${i}" tabindex="-1">${esc(x.label)}${x.sub?`<small>${esc(x.sub)}</small>`:''}</button>`).join('')||'';
    nodes=[...el.children];layout();
  }
  function layout(){
    const c=Math.round(pos);
    nodes.forEach((n,i)=>{
      const th=(i-pos)*o.step,r=th*Math.PI/180;
      if(Math.abs(th)>88){n.style.visibility='hidden';return;}
      const cos=Math.cos(r),off=o.radius*Math.sin(r);
      n.style.visibility='';
      n.style.opacity=Math.max(0,Math.pow(cos,2.2)).toFixed(3);
      n.style.transform=o.axis==='x'
        ?`translate(-50%,-50%) translateX(${off}px) rotateY(${-th*.75}deg) scale(${(.62+.38*cos).toFixed(3)})`
        :`translate(-50%,-50%) translateY(${off}px) rotateX(${th*.75}deg) scale(${(.8+.2*cos).toFixed(3)})`;
      n.classList.toggle('on',i===c);
      n.setAttribute('aria-selected',i===c);
    });
    if(c!==last&&c>=0&&c<o.items.length){last=c;tick(o.pitch);}
  }
  function clamp(p){const max=o.items.length-1;return p<0?p*.35:p>max?max+(p-max)*.35:p;}
  function to(i,instant){
    cancelAnimationFrame(anim);i=Math.max(0,Math.min(o.items.length-1,i));
    const from=pos,dist=i-from,dur=reduce||instant?0:Math.min(720,220+70*Math.abs(dist)),t0=performance.now();
    const step=now=>{const k=dur?Math.min(1,(now-t0)/dur):1,e=1-Math.pow(1-k,3);pos=from+dist*e;layout();if(k<1)anim=requestAnimationFrame(step);else{pos=i;layout();o.onSettle?.(i);}};
    anim=requestAnimationFrame(step);
  }
  let drag=null;
  el.addEventListener('pointerdown',e=>{if(!o.items.length)return;cancelAnimationFrame(anim);drag={x:e.clientX,y:e.clientY,p:pos,moved:0,v:0,t:performance.now(),q:o.axis==='x'?e.clientX:e.clientY};el.setPointerCapture(e.pointerId);});
  el.addEventListener('pointermove',e=>{if(!drag)return;const q=o.axis==='x'?e.clientX:e.clientY,now=performance.now();
    drag.moved=Math.max(drag.moved,Math.hypot(e.clientX-drag.x,e.clientY-drag.y));
    const d=(q-(o.axis==='x'?drag.x:drag.y))/pxPer();
    drag.v=.8*drag.v+.2*((q-drag.q)/pxPer())/Math.max(1,now-drag.t);drag.q=q;drag.t=now;
    pos=clamp(drag.p-d);layout();});
  const end=e=>{if(!drag)return;const d=drag;drag=null;
    if(d.moved<6){const hit=document.elementFromPoint(e.clientX,e.clientY)?.closest('.wheel-item');if(hit&&el.contains(hit))return to(+hit.dataset.i);return to(Math.round(pos));}
    const fling=Math.max(-4,Math.min(4,d.v*60));to(Math.round(pos-(performance.now()-d.t>80?0:fling)));};
  el.addEventListener('pointerup',end);el.addEventListener('pointercancel',end);
  el.addEventListener('wheel',e=>{e.preventDefault();cancelAnimationFrame(anim);const d=(Math.abs(e.deltaX)>Math.abs(e.deltaY)?e.deltaX:e.deltaY)/pxPer();pos=clamp(pos+d);layout();clearTimeout(el._w);el._w=setTimeout(()=>to(Math.round(pos)),120);},{passive:false});
  el.addEventListener('keydown',e=>{const back=o.axis==='x'?'ArrowLeft':'ArrowUp',fwd=o.axis==='x'?'ArrowRight':'ArrowDown';if(e.key===back){e.preventDefault();to(Math.round(pos)-1);}if(e.key===fwd){e.preventDefault();to(Math.round(pos)+1);}});
  build();
  return {to,set(items,i){o.items=items;pos=last=Math.max(0,Math.min(items.length-1,i));build();},get index(){return Math.round(pos);},stop(){cancelAnimationFrame(anim);}};
}

/* ---------- The synthetic wall ---------- */
const HOLDS=[[64,92,'#e3c04a',16,13],[118,226,'#3bb59b',22,15],[306,188,'#d4607e',23,16],[350,280,'#8a78d6',17,13],[92,418,'#e3c04a',18,15],[318,462,'#d8d6d0',18,16],
  [156,588,'#d4607e',16,12],[352,632,'#3bb59b',15,13],[112,742,'#8a78d6',25,15],[318,826,'#e3c04a',22,15],[176,912,'#3bb59b',19,15],[262,1022,'#8a78d6',23,15],
  [340,988,'#d8d6d0',17,15],[236,152,'#3bb59b',14,11],[48,300,'#d4607e',13,11],[262,712,'#e3c04a',14,12],[52,560,'#3bb59b',14,12],[214,834,'#d8d6d0',12,10]];
function climber(dx,dy){
  const S=[200+dx,452+dy],H=[200+dx,532+dy],P=(x,y)=>`${x+dx},${y+dy}`;
  return `<g class="cam-climber">
   ${[[166,388],[240,392],[178,656],[232,610]].map(([x,y])=>`<ellipse cx="${x+dx}" cy="${y+dy}" rx="20" ry="14" fill="#3bb59b" transform="rotate(-12 ${x+dx} ${y+dy})"/>`).join('')}
   <g fill="none" stroke="#121417" stroke-width="10" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="${S} ${P(180,428)} ${P(164,384)}"/><polyline points="${S} ${P(222,420)} ${P(240,386)}"/>
    <line x1="${S[0]}" y1="${S[1]-6}" x2="${H[0]}" y2="${H[1]}"/>
    <polyline points="${H} ${P(186,580)} ${P(178,652)}"/><polyline points="${H} ${P(222,566)} ${P(232,606)}"/>
   </g><circle cx="${206+dx}" cy="${424+dy}" r="15" fill="#121417"/></g>`;
}
function wall(scene){
  const boulder=scene==='抱石',dy=boulder?40:0;
  const holds=HOLDS.filter(h=>!boulder||(h[1]>250&&h[1]<870)).map(([x,y,c,rx,ry],i)=>`<ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${ry}" fill="${c}" opacity=".9" transform="rotate(${(i*37)%60-30} ${x} ${y})"/>`).join('');
  const rope=scene==='顶绳'
    ?`<circle cx="200" cy="44" r="9" fill="none" stroke="#9aa0a8" stroke-width="4"/><rect x="186" y="26" width="28" height="6" rx="3" fill="#9aa0a8"/>
      <g class="cam-rope" fill="none" stroke-linecap="round"><path d="M200 52 C198 220 206 360 202 520" stroke="#ff8a45" stroke-width="4"/><path d="M200 52 C192 400 214 760 186 1200" stroke="#b9653a" stroke-width="4"/></g>`
    :scene==='先锋'
    ?`${[[196,1010,1],[208,860,1],[192,700,1],[206,290,0],[194,150,0]].map(([x,y,c])=>`<g transform="translate(${x} ${y})" opacity="${c?1:.75}"><rect x="-3" y="0" width="6" height="22" rx="3" fill="#2b2e33"/><circle cx="0" cy="-4" r="6" fill="none" stroke="#b8bec6" stroke-width="3"/><ellipse cx="0" cy="30" rx="6" ry="9" fill="none" stroke="#c9ced5" stroke-width="3"/></g>`).join('')}
      <path class="cam-rope" d="M150 1200 C180 1100 196 1060 196 1040 S210 900 208 890 S190 740 192 730 S204 600 200 540" fill="none" stroke="#ff8a45" stroke-width="4" stroke-linecap="round"/>`
    :`<rect x="0" y="0" width="400" height="230" fill="#17191d"/><rect x="0" y="226" width="400" height="6" fill="#5a5f68"/>
      <g><rect x="-10" y="880" width="210" height="330" rx="14" fill="#2f4a6a"/><rect x="204" y="880" width="210" height="330" rx="14" fill="#2b4562"/><rect x="0" y="874" width="400" height="10" fill="#39577a" opacity=".6"/></g>`;
  return `<svg class="cam-wall" viewBox="0 120 400 900" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
   <defs><linearGradient id="cam-g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2f3238"/><stop offset=".55" stop-color="#383c43"/><stop offset="1" stop-color="#1b1d21"/></linearGradient></defs>
   <rect width="400" height="1200" fill="url(#cam-g)"/>
   <g stroke="#ffffff0d" stroke-width="1.5">${[133,266].map(x=>`<line x1="${x}" y1="0" x2="${x}" y2="1200"/>`).join('')}${[300,600,900].map(y=>`<line x1="0" y1="${y}" x2="400" y2="${y}"/>`).join('')}</g>
   ${holds}${rope}${climber(0,dy)}<g class="cam-frame" transform="translate(0 ${dy})" fill="none" stroke-width="3" stroke-linecap="round">${[[128,350,1,1],[272,350,-1,1],[128,690,1,-1],[272,690,-1,-1]].map(([x,y,a,b])=>`<path d="M${x} ${y+22*b}V${y}H${x+22*a}"/>`).join('')}</g></svg>`;
}

/* ---------- Markup ---------- */
const icons={
  pin:'<svg viewBox="0 0 24 24"><path d="M12 21s-6.5-5.6-6.5-11a6.5 6.5 0 0 1 13 0c0 5.4-6.5 11-6.5 11z"/><circle cx="12" cy="10" r="2.4"/></svg>',
  down:'<svg viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></svg>',
  close:'<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  flip:'<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 0 0-14.3-4.9L4 8"/><path d="M4 4v4h4"/><path d="M4 13a8 8 0 0 0 14.3 4.9L20 16"/><path d="M20 20v-4h-4"/></svg>',
  search:'<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/></svg>',
  scan:'<svg viewBox="0 0 24 24"><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3M4 12h16"/></svg>',
  check:'<svg viewBox="0 0 24 24"><path d="m5 12 5 5 9-10"/></svg>'
};
function shutter(rec){
  return `<svg viewBox="0 0 64 64" aria-hidden="true"><path class="jug" d="M12 33c-1-12 7-22 20-23 11-1 20 6 21 16 .6 5-1.6 8-1 12 .8 7-5 15-17 16C21 55 13 45 12 33z"/>
    ${rec?'<rect class="hole" x="26" y="27" width="12" height="12" rx="3"/>':'<circle class="hole" cx="33" cy="33" r="6"/><circle cx="33" cy="33" r="2.2" fill="#ffffff22"/>'}</svg>`;
}
function view(recording){
  const d=cam.discipline,gl=gradeOf(),latest=getRecords()[0];
  const gymLabel=cam.gym?esc(cam.gym.name):'正在定位…';
  const gymSub=cam.gym?esc(cam.gymNote):'附近的岩馆';
  return `<div class="cam ${recording?'is-rec':''} ${cam.front?'is-front':''}" style="--cam-accent:${ACCENTS[cam.accent]}">
   <div class="cam-view">${wall(cam.scene)}<div class="cam-scan"></div></div>
   <div class="cam-top">
     <button class="cam-icon cam-close" data-cam="close" aria-label="关闭拍摄">${icons.close}</button>
     <button class="cam-pill cam-gym" data-cam="gym" aria-label="选择岩馆：${gymLabel}">${icons.pin}<span><small>${gymSub}</small><b>${gymLabel}</b></span>${icons.down}</button>
     <button class="cam-pill cam-disc ${cam.scanning?'scanning':''}" data-cam="menu" aria-haspopup="menu" aria-expanded="${cam.menu}" aria-label="攀爬方式">
       ${cam.scanning?'<span class="cam-dots"><i></i><i></i><i></i></span><b>识别中</b>':`<span class="cam-badge ${cam.mode}">${cam.mode==='auto'?'AUTO':'手动'}</span><b>${d}</b>`}</button>
     ${cam.menu?`<div class="cam-menu" role="menu">
        <button role="menuitemradio" aria-checked="${cam.mode==='auto'}" data-cam="auto">${icons.scan}<span>自动识别<small>重新扫一次画面</small></span>${cam.mode==='auto'?icons.check:''}</button>
        ${DISCIPLINES.map(x=>`<button role="menuitemradio" aria-checked="${cam.mode==='manual'&&d===x}" data-cam="pick:${x}"><span>${x}<small>${x==='抱石'?'V 等级':'YDS'}</small></span>${cam.mode==='manual'&&d===x?icons.check:''}</button>`).join('')}
      </div>`:''}
   </div>
   <div class="cam-hint" role="status"></div>
   <div class="cam-recbar" role="timer"><i></i><b id="cam-clock">00:00</b><span>${esc(d||'')} · ${esc(gl)} · ${gymLabel}</span></div>
   <div class="cam-bottom">
     <div class="cam-grade-label">难度 <span>·</span> ${scale(d||cam.scene)==='抱石'?'V 等级':'YDS'}</div>
     <div class="cam-pointer"></div>
     <div class="cam-wheel" id="cam-grade" role="listbox" tabindex="0" aria-label="难度"></div>
     <div class="cam-controls">
       <button class="cam-album" data-cam="album" aria-label="攀爬册">${latest?`<span>${esc(latest.grade==='未标级'?'—':latest.grade)}</span>`:'<svg viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="14" rx="2.5"/><path d="m4 16 4.5-4.5 4 4 2.5-2.5L20 18"/></svg>'}</button>
       <button class="cam-shutter" data-cam="${recording?'stop':'start'}" aria-label="${recording?'停止并保存':'开始录制'}">${shutter(recording)}</button>
       <button class="cam-icon cam-flip" data-cam="flip" aria-label="翻转镜头">${icons.flip}</button>
     </div>
   </div>
   ${cam.sheet?gymSheet():''}
  </div>`;
}
function gymList(){
  const q=cam.query.trim();
  return [...GYMS,...cam.custom].filter(g=>(!q||g.name.includes(q))&&(cam.filter==='全部'||(cam.filter==='3 km 内'?g.km!=null&&g.km<=3:g.kinds.includes(cam.filter))));
}
function gymSheet(){
  return `<div class="cam-scrim" data-cam="sheet-close"></div><section class="cam-sheet" role="dialog" aria-label="选择岩馆">
    <div class="cam-sheet-head"><b>选择岩馆</b><button data-cam="sheet-close">完成</button></div>
    <label class="cam-search">${icons.search}<input id="cam-query" placeholder="搜索岩馆名称" value="${esc(cam.query)}" autocomplete="off"></label>
    <div class="cam-filters">${['全部','抱石','绳索','3 km 内'].map(x=>`<button class="${cam.filter===x?'on':''}" data-cam="filter:${x}" aria-pressed="${cam.filter===x}">${x}</button>`).join('')}</div>
    <div class="cam-drum-wrap"><div class="cam-drum-band"></div><div class="cam-drum" id="cam-drum" role="listbox" tabindex="0" aria-label="岩馆"></div><div class="cam-empty" id="cam-empty"></div></div>
    <p class="cam-note">按距离排序只在本机进行，位置不保存。距离是约数。</p>
    <div class="cam-sheet-grip"></div></section>`;
}

/* ---------- Behaviour ---------- */
function hint(text,ms=3200,sub=''){const h=$('.cam-hint');if(!h)return;h.innerHTML=esc(text)+(sub?`<small>${esc(sub)}</small>`:'');h.classList.add('show');clearTimeout(h._t);h._t=setTimeout(()=>h.classList.remove('show'),ms);}
function scan(){
  cam.scanning=true;cam.discipline=cam.discipline||null;paint();
  later(()=>{cam.scanning=false;cam.discipline=cam.scene;paint();hint(REASON[cam.scene],3600,'可能认错，点右上角可以改');},1800);
}
function locate(){
  if(cam.gym)return;
  later(()=>{cam.gym=GYMS[0];cam.gymNote='最近 · '+km(GYMS[0]);paint();},700);
}
function mountWheels(){
  cam.wheels.forEach(w=>w.stop());cam.wheels=[];
  const g=$('#cam-grade');
  if(g){const list=gradeOptions(scale(cam.discipline||cam.scene)),cur=list.indexOf(gradeOf());
    const w=wheel(g,{items:list.map(x=>({label:x})),index:cur<0?0:cur,axis:'x',step:25,radius:200,pitch:1,
      onSettle:i=>{cam.grade[scale(cam.discipline||cam.scene)]=list[i];const bar=$('.cam-recbar span');if(bar)bar.textContent=`${cam.discipline} · ${list[i]} · ${cam.gym?.name||''}`;}});
    cam.wheels.push(w);}
  const dr=$('#cam-drum');
  if(dr){const items=gymList(),cur=Math.max(0,items.indexOf(cam.gym));
    $('#cam-empty').innerHTML=items.length?'':`<p>没有找到「${esc(cam.query)}」</p>${cam.query.trim()?`<button data-cam="custom">把「${esc(cam.query.trim())}」作为自定义岩馆</button>`:''}`;
    const w=wheel(dr,{items:items.map(g=>({label:g.name,sub:km(g)})),index:cur,axis:'y',step:24,radius:96,pitch:.62,
      onSettle:i=>{const g=items[i];if(!g||g===cam.gym)return;cam.gym=g;cam.gymNote=g===GYMS[0]?'最近 · '+km(g):km(g);const b=$('.cam-gym');if(b){b.querySelector('small').textContent=cam.gymNote;b.querySelector('b').textContent=g.name;}}});
    cam.wheels.push(w);}
}
function paint(){
  const screen=document.getElementById('native-screen');if(!screen.querySelector('.cam'))return;
  const focus=document.activeElement?.id;screen.innerHTML=view(state.page==='recording');mountWheels();
  if(focus==='cam-query'){const q=$('#cam-query');q.focus();q.setSelectionRange(q.value.length,q.value.length);}
  if(state.page==='recording')clock();
}
function clock(){
  const el=$('#cam-clock');if(!el)return;const s=Math.floor((Date.now()-cam.startedAt)/1000);
  el.textContent=`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
}
function stop(){
  const now=new Date(),d=cam.discipline||cam.scene,g=gradeOf();
  addRecord({title:'这一趟',grade:g,sent:false,day:now.toISOString().slice(0,10),time:now.toTimeString().slice(0,5),gym:cam.gym?.name||'',folder:'',discipline:d,url:null,result:false,plan:false});
  state.page='capture';render();
  toast(`已存进攀爬册 · ${d} · ${g} · ${cam.gym?.name||'未选岩馆'}`);
}
function onClick(e){
  const b=e.target.closest('[data-cam]');if(!b||!b.closest('.cam')){if(cam.menu&&e.target.closest('.cam')&&!e.target.closest('.cam-menu')){cam.menu=false;paint();}return;}
  e.stopPropagation();const a=b.dataset.cam,[k,v]=a.split(':');
  if(a!=='menu'&&!a.startsWith('pick')&&a!=='auto')cam.menu=false;
  switch(k){
    case'close':return go(state.page==='recording'?'capture':'journal');
    case'gym':cam.sheet=true;cam.query='';return paint();
    case'sheet-close':cam.sheet=false;if(state.page==='venues'){state.page='capture';return render();}return paint();
    case'filter':cam.filter=v;return paint();
    case'custom':{const name=cam.query.trim();if(!name)return;const g={name,km:null,kinds:['抱石','绳索']};cam.custom.push(g);cam.gym=g;cam.gymNote='自定义';cam.query='';cam.filter='全部';cam.sheet=false;tick(.62);return paint();}
    case'menu':if(cam.scanning)return;cam.menu=!cam.menu;return paint();
    case'auto':cam.menu=false;cam.mode='auto';return scan();
    case'pick':{const old=cam.discipline;cam.menu=false;cam.mode='manual';cam.discipline=v;paint();if(scale(old)!==scale(v))hint(v==='抱石'?'换成 V 等级':'换成 YDS',1600);return;}
    case'album':return go('journal');
    case'flip':cam.front=!cam.front;tick(.8);paint();return hint(cam.front?'前置镜头 · 原型只示意':'后置镜头',1400);
    case'start':if(cam.scanning)return hint('等识别完再开始',1400);cam.startedAt=Date.now();tick(1.3);return go('recording');
    case'stop':tick(1.3);return stop();
  }
  paint();
}
function onInput(e){if(e.target.id==='cam-query'){cam.query=e.target.value;paint();}}

/* ---------- Entry points for app.js ---------- */
let bound=false,recTimer=0,wasCamera=false;
function mount(page){
  const isCam=['capture','recording','venues'].includes(page);
  clearInterval(recTimer);cam.wheels.forEach(w=>w.stop());cam.wheels=[];
  if(!isCam){cam.timers.forEach(clearTimeout);cam.timers=[];cam.menu=false;cam.sheet=false;cam.scanning=false;wasCamera=false;return;}
  if(!bound){const s=document.getElementById('native-screen');s.addEventListener('click',onClick,true);s.addEventListener('input',onInput);bound=true;}
  if(page==='venues')cam.sheet=true;
  if(page==='recording'){if(!cam.startedAt)cam.startedAt=Date.now();if(!cam.discipline)cam.discipline=cam.scene;recTimer=setInterval(clock,500);}
  else cam.startedAt=0;
  mountWheels();
  if(!wasCamera){wasCamera=true;locate();if(cam.mode==='auto')scan();}
  if(page==='recording')clock();
}
function controls(){
  return `<label>模拟画面<select id="cam-scene">${DISCIPLINES.map(x=>`<option ${cam.scene===x?'selected':''}>${x}</option>`).join('')}</select></label>
   <label>强调色<select id="cam-accent">${Object.keys(ACCENTS).map(x=>`<option ${cam.accent===x?'selected':''}>${x}</option>`).join('')}</select></label>
   <label class="cam-sound-toggle"><input type="checkbox" id="cam-sound" ${cam.sound?'checked':''}> 轮盘音效</label>
   <p class="small">浏览器要先有一次点击才能出声。识别结果、岩馆与距离都是示意数据。</p>`;
}
document.addEventListener('change',e=>{
  if(e.target.id==='cam-scene'){cam.scene=e.target.value;if(cam.mode==='auto')scan();else paint();}
  if(e.target.id==='cam-accent'){cam.accent=e.target.value;paint();}
  if(e.target.id==='cam-sound'){cam.sound=e.target.checked;}
});
return {render:view,mount,controls};
};
