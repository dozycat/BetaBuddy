/* Local preview adapter. The app's real scene and all its controls remain mounted. */
(()=>{
 let access={platform:'ios',member:false,memberVenues:[]};
 const locked=()=>access.platform==='ios'&&!access.member;
 function decorate(){
  const cat=document.querySelector('#species [data-sp="cat"]');
  if(cat){cat.textContent=locked()?'猫 · 会员':'猫';cat.setAttribute('aria-label',cat.textContent);}
  document.querySelectorAll('#venuelist button[data-venue]').forEach(b=>{
   b.querySelector('[data-preview-member]')?.remove();
   if(locked()&&access.memberVenues.includes(b.dataset.venue)){const label=document.createElement('span');label.dataset.previewMember='';label.textContent='会员';b.append(label);}
  });
 }
 const tell=(type)=>parent.postMessage({source:'betabuddy-preview-gym',type},location.origin);
 window.webkit={messageHandlers:{gym:{postMessage:(m)=>{if(['ready','openSettings','hintSeen','error','requestCat'].includes(m.type))tell(m.type);}}}};
 addEventListener('message',e=>{
  if(e.origin!==location.origin||e.source!==parent||e.data?.source!=='betabuddy-preview')return;
  if(e.data.type==='access'){access=e.data.access;decorate();}
  if(e.data.type==='panel')window.BetaBuddyGym?.debug({sheet:e.data.sheet,tab:e.data.tab});
 });
 addEventListener('DOMContentLoaded',()=>{
  // Presentation-only backdrop; all meshes and their real materials stay intact.
  if(typeof scene!=='undefined'&&scene.background&&typeof scene.background.set==='function'){scene.background.set(0xedf7fc);if(typeof invalidate==='function')invalidate();}
  // Same non-member species gate as GymSession.catMembershipScript.
  if(typeof window.setSpecies==='function'){
   const original=window.setSpecies;
   window.setSpecies=(species,options)=>{if(species==='cat'&&locked()){tell('requestCat');return false;}return original(species,options);};
  }
  document.addEventListener('click',e=>{const b=e.target.closest('[data-venue]');if(b&&locked()&&access.memberVenues.includes(b.dataset.venue)){e.preventDefault();e.stopImmediatePropagation();tell('requestCat');}},true);
  decorate();
  // Route names, walls, presets and computed values are never fabricated or re-written here.
 });
})();
