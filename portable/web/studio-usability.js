/* Persistent local controls. All progress comes from the existing task API. */
(() => {
"use strict";
const $=id=>document.getElementById(id);
// Keep the dock's essential styles with its script: an older cached page theme
// must never turn the floating controls into an unstyled document footer.
const dockStyle=document.createElement('style');dockStyle.id='studio-floating-dock-style';
dockStyle.textContent=`
body.has-generation-dock{padding-bottom:var(--studio-dock-space,140px)!important}
#studio-generation-dock{position:fixed!important;z-index:1000!important;left:50%!important;right:auto!important;bottom:max(16px,env(safe-area-inset-bottom))!important;transform:translateX(-50%)!important;width:min(calc(100vw - 40px),1400px)!important;box-sizing:border-box!important;margin:0!important;padding:14px 16px!important;border:1px solid rgba(238,138,178,.48)!important;border-radius:19px!important;background:rgba(255,252,254,.97)!important;box-shadow:0 12px 38px rgba(116,58,87,.19),0 2px 8px rgba(237,90,148,.09)!important;backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);font-family:Segoe UI,Microsoft YaHei UI,sans-serif!important;color:#3c3448;font-size:13px;line-height:1.45}
#studio-generation-dock *{box-sizing:border-box}
#studio-generation-dock .studio-dock-inner{max-width:none;margin:0;display:flex;align-items:center;flex-wrap:wrap;gap:10px}
#studio-generation-dock .studio-dock-status{flex:1 1 260px;min-width:190px;display:grid;gap:6px;overflow:hidden;padding:0 12px 0 4px}
#studio-generation-dock .studio-dock-status strong{display:flex;align-items:center;gap:8px;margin:0;font-size:14px;font-weight:700;color:#4c3c52;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#studio-generation-dock .studio-dock-status strong:before{content:'';flex:0 0 7px;width:7px;height:7px;border-radius:50%;background:#ee82ac;box-shadow:0 0 0 4px #fff0f6}
#studio-generation-dock .studio-dock-status span{font-size:11px;color:#927f93;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-height:none}
#studio-generation-dock progress{display:block;appearance:none;-webkit-appearance:none;width:100%;height:5px;border:0;border-radius:10px;overflow:hidden;background:#f6e8f0;accent-color:#ed609b}
#studio-generation-dock progress::-webkit-progress-bar{background:#f6e8f0;border-radius:10px}
#studio-generation-dock progress::-webkit-progress-value{background:linear-gradient(90deg,#f39abb,#e84c91);border-radius:10px}
#studio-generation-dock progress::-moz-progress-bar{background:#e95b96;border-radius:10px}
#studio-generation-dock #dock-count-label{display:flex;align-items:center;gap:8px;flex:0 0 120px;height:52px;min-width:0;margin:0;padding:8px 10px;background:#fff8fc;border:1px solid #f1dfe9;border-radius:12px;color:#94738a;font-size:11px;font-weight:500}#studio-generation-dock #dock-count-label span{white-space:nowrap}
#studio-generation-dock #dock-takes{display:block;width:42px;min-width:0;height:34px;border:0;border-radius:6px;padding:2px;margin:0;background:transparent;text-align:center;color:#78506a;font:600 18px Segoe UI,sans-serif;box-shadow:none;outline:none}
#studio-generation-dock button,#studio-generation-dock summary{display:inline-flex;align-items:center;justify-content:center;gap:4px;flex:0 0 auto;height:52px;min-height:52px;margin:0;padding:0 15px;border:1px solid #efdfeb;border-radius:12px;background:#fff;color:#8e6682;font:600 12px Segoe UI,Microsoft YaHei UI,sans-serif;text-decoration:none;white-space:nowrap;cursor:pointer;box-shadow:none}
#studio-generation-dock button:hover,#studio-generation-dock summary:hover{border-color:#eaa6c4;background:#fff5fa}
#studio-generation-dock #dock-generate{min-width:184px;padding:0 25px;border:0;background:linear-gradient(115deg,#f180ab,#e4488c);box-shadow:0 5px 13px #e66c9b26;color:white;font-size:14px;font-weight:750}
#studio-generation-dock #dock-cancel{min-width:92px;color:#b17b96;background:#fff9fc}
#studio-generation-dock button:disabled{opacity:.42;cursor:default;box-shadow:none}
#studio-generation-dock #dock-film{min-width:99px;flex-direction:column;gap:1px}
#studio-generation-dock #dock-film-count{font-size:10px;font-weight:400;color:#b08da4}
#studio-generation-dock #dock-more{position:relative;margin:0;padding:0;border:0}
#studio-generation-dock #dock-more summary{width:50px;padding:0;font-size:19px;list-style:none}
#studio-generation-dock #dock-more summary::-webkit-details-marker{display:none}
#studio-generation-dock .studio-dock-menu{position:absolute;bottom:64px;right:0;min-width:180px;display:grid;gap:6px;padding:9px;border:1px solid #efd6e4;border-radius:14px;background:#fffafd;box-shadow:0 10px 35px #67445820}
#studio-generation-dock .studio-dock-menu button{width:100%;height:40px;min-height:40px;justify-content:flex-start;border:0;background:transparent}
#studio-generation-dock [hidden]{display:none!important}
@media(max-width:1050px){#studio-generation-dock .studio-dock-status{flex-basis:210px}#studio-generation-dock #dock-generate{min-width:158px}#studio-generation-dock button{padding:0 11px}}
@media(max-width:900px){#studio-generation-dock .studio-dock-status{flex:1 0 100%;padding:0 3px 3px}#studio-generation-dock .studio-dock-inner{gap:8px}#studio-generation-dock{padding:12px!important}#studio-generation-dock #dock-generate{flex:1 1 160px}#studio-generation-dock #dock-count-label{flex-basis:104px}}
@media(max-width:540px){#studio-generation-dock{width:calc(100vw - 20px)!important;bottom:10px!important}#studio-generation-dock button{height:44px;min-height:44px}#studio-generation-dock #dock-count-label{height:44px}#studio-generation-dock #dock-more summary{height:44px;min-height:44px}}
`;
document.head.append(dockStyle);
const dock=document.createElement("section");dock.id="studio-generation-dock";dock.setAttribute("aria-label","常驻生成控制条");
dock.innerHTML=`<div class="studio-dock-inner">
<div class="studio-dock-status"><strong id="dock-status" role="status" aria-live="polite">就绪</strong><span id="dock-detail">设置候选数量后开始生成</span><progress id="dock-progress" max="1" value="0" aria-label="当前任务真实进度"></progress></div>
<label id="dock-count-label"><span>每句候选</span><input id="dock-takes" type="number" min="1" max="32" step="1" value="3" aria-label="每句候选数量"></label>
<button id="dock-generate" type="button" class="primary">生成 / 补齐候选</button><button id="dock-cancel" type="button" class="secondary" disabled>停止生成</button>
<button id="dock-film" type="button" class="secondary"><span>成片选片</span><small id="dock-film-count"></small></button><button id="dock-model-paths" type="button" class="secondary">模型路径</button><details id="dock-more"><summary aria-label="更多操作" title="更多操作">···</summary><div class="studio-dock-menu"><button id="dock-history" type="button">任务与结果</button><button id="dock-social" type="button">By T8star · 社媒链接 ↗</button></div></details></div>`;
document.body.append(dock);document.body.classList.add("has-generation-dock");
const reserveSpace=()=>document.body.style.setProperty('--studio-dock-space',Math.ceil(dock.getBoundingClientRect().height)+48+'px');
if(window.ResizeObserver)new ResizeObserver(reserveSpace).observe(dock);reserveSpace();
let jobs=window.latestStudioJobs||[],current=null,lastProjectId=null;
const names={creative_batch:"作品演出",creative_reference:"参考编码",creative_variation:"改稿 / 局部重配",creative_quality:"转写质检",creative_recording:"整理录音",creative_delivery:"制作交付包",creative_benchmark:"本机性能对照",configure:"应用模型与运行设置",tts:"语音生成",generate:"语音生成",transcribe:"本地音频转写",preview:"演出试听解码",align:"词级对齐",scene:"多人场景",long:"长文生成",revise:"导演改稿",infill:"局部重配",unload:"释放内存"};
function creative(){return document.querySelector('#tabs [data-page="creative"]')?.classList.contains("active");}
function announce(text){if(typeof window.notice==="function")window.notice(text);else if(typeof notice==="function")notice(text);}
function refresh(){
 const cd=window.CreativeDesk,projectId=cd?.state.project?.project_id;
 const merged=new Map(jobs.map(job=>[job.id,job]));for(const job of cd?.state.jobs?.values()||[])if(!merged.has(job.id))merged.set(job.id,job);
 const ordered=[...merged.values()].sort((a,b)=>(Number(b.created)||0)-(Number(a.created)||0));
 const running=ordered.find(j=>j.status==="running"),queued=ordered.filter(j=>j.status==="queued");current=running||queued[0]||null;
 const label=$('dock-status'),detail=$('dock-detail'),progress=$('dock-progress');
 if(current){const name=names[current.payload?.mode]||"本地任务",value=Number(current.progress);label.textContent=(running?"正在处理 · ":"排队中 · ")+name;
  if(running&&Number.isFinite(value)&&value>0){progress.value=Math.min(1,Math.max(0,value));detail.textContent=(value*100).toFixed(1)+"%"+(queued.length?" · 另有 "+queued.length+" 项排队":"");}
  else{progress.removeAttribute("value");detail.textContent=running?"正在处理 · 等待本地引擎回报进度":queued.length+" 项等待执行";}
  if(current.cancelable===false)detail.textContent+=" · 正在保存结果";
 }else{const latest=ordered[0];label.textContent=latest?({done:"上次任务已完成",failed:"上次任务失败",cancelled:"上次任务已取消",interrupted:"上次任务已中断"}[latest.status]||"就绪"):"就绪";
  detail.textContent=latest?.error||(creative()?(cd?.state.project?.name||"选择角色，填写台词，即可开始创作"):"选择参考与台词，准备这一遍演出");progress.value=latest?.status==="done"?1:0;}
 $('dock-cancel').disabled=!current||current.cancelable===false;
 const activeCreative=creative();$('dock-count-label').hidden=!activeCreative;
 const submitted=window.studioSubmitFeedback;if(!activeCreative&&submitted?.type==='error'){label.textContent='请检查输入';detail.textContent=submitted.message;}
 const error=document.getElementById('cd-message');if(activeCreative&&error&&!error.hidden&&error.classList.contains('error'))detail.textContent=error.textContent;
 $('dock-film').hidden=!activeCreative;$('dock-film-count').textContent=$('cd-film-count')?.textContent||'';
 if(activeCreative){const source=$('cd-takes');if(source&&document.activeElement!==$('dock-takes'))$('dock-takes').value=source.value;
  const target=$('cd-run-all'),pending=ordered.some(j=>['queued','running'].includes(j.status)&&j.payload?.project_id&&j.payload.project_id===projectId);
  $('dock-generate').textContent="生成 / 补齐候选";$('dock-generate').disabled=!!target?.disabled||pending;
 }else{const auxiliary=['history','environment','voices'].some(p=>document.querySelector(`#tabs [data-page="${p}"]`)?.classList.contains('active'));
  $('dock-generate').textContent=auxiliary?"返回创作":$('generate')?.textContent||"开始生成";$('dock-generate').disabled=!auxiliary&&!!$('generate')?.disabled;}
 if(lastProjectId!==projectId){lastProjectId=projectId;}
}
window.updateStudioGenerationBar=value=>{if(Array.isArray(value))jobs=value;refresh();};
window.refreshStudioDock=refresh;
$('dock-takes').addEventListener('input',()=>{const n=Number($('dock-takes').value);if(!Number.isInteger(n)||n<1||n>32)return;const source=$('cd-takes');if(source){source.value=String(n);source.dispatchEvent(new Event('input',{bubbles:true}));}});
$('dock-generate').onclick=()=>{if(creative()){const n=Number($('dock-takes').value);if(!Number.isInteger(n)||n<1||n>32){announce('候选数量请输入 1–32 的整数。');return;}$('cd-run-all').click();}
 else if($('dock-generate').textContent==='返回创作')window.CreativeDesk?.open('projects');else $('generate')?.click();};
$('dock-cancel').onclick=async()=>{if(!current)return;const id=current.id;$('dock-cancel').disabled=true;try{const response=await fetch('/api/jobs/'+encodeURIComponent(id)+'/cancel',{method:'POST'});if(!response.ok)throw Error('取消未成功，请查看任务状态');announce('取消请求已提交，已完成片段保留。');}catch(e){announce(e.message)}finally{$('dock-cancel').disabled=false;}};
$('dock-model-paths').onclick=()=>{if(typeof showPage==='function')showPage('environment');refresh();const panel=document.querySelector('.runtime-form');panel?.scrollIntoView({behavior:'smooth',block:'start'});$('runtime-model')?.focus({preventScroll:true});};
$('dock-film').onclick=()=>{window.CreativeDesk?.open('projects');$('cd-film-section')?.scrollIntoView({behavior:'smooth',block:'start'});};
$('dock-history').onclick=()=>{if(typeof showPage==='function')showPage('history');refresh();$('history-panel')?.scrollIntoView({behavior:'smooth',block:'start'});};
$('dock-social').onclick=()=>{const footer=$('studio-social-footer');footer?.scrollIntoView({behavior:'smooth',block:'center'});footer?.classList.add('social-highlight');setTimeout(()=>footer?.classList.remove('social-highlight'),1800);};
window.deleteStudioHistoryItem=async job=>{if(!confirm('删除这条生成记录？可在创作工程的回收站恢复，音频暂存。'))return;try{const response=await fetch('/api/creative/catalog/job/'+job.id,{method:'DELETE',headers:{'Content-Type':'application/json'},body:'{}'});if(!response.ok){const body=await response.json();throw Error(body.detail||'删除失败');}
 if(typeof lastShown!=='undefined'&&lastShown===job.id){$('audio').pause();$('audio').removeAttribute('src');$('audio').load();$('output-text').textContent='记录已删除，可到回收站恢复。';$('output-links').hidden=true;restoreLatestAudio=false;lastShown=null;}
 window.CreativeDesk?.state.jobs.delete(job.id);jobs=jobs.filter(j=>j.id!==job.id);if(typeof refreshJobs==='function')await refreshJobs();announce('记录已删除，可在回收站恢复。');refresh();}catch(e){announce(e.message)}};
const deleteResult=document.createElement('button');deleteResult.type='button';deleteResult.className='ghost danger';deleteResult.textContent='删除这次生成';deleteResult.onclick=()=>{const job=jobs.find(j=>typeof lastShown!=='undefined'&&j.id===lastShown);if(job)window.deleteStudioHistoryItem(job);};$('output-links')?.append(deleteResult);
// Export existing refresh helpers for recovery controls, without replacing handlers.
if(typeof refreshJobs==='function')window.refreshJobs=refreshJobs;
$('tabs')?.addEventListener('click',()=>queueMicrotask(refresh));
setInterval(refresh,500);refresh();
})();
