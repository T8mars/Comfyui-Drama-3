const toolsPanel = document.createElement('div');
toolsPanel.className = 'take-tools';
toolsPanel.innerHTML = `<label>会话中的演出版本<select id="take-index"><option value="-1">最新版本</option></select></label>
  <button type="button" class="secondary" id="preview-take">试听所选版本</button>
  <section id="source-preview" hidden style="padding:12px;border:1px solid #f0d4e2;border-radius:12px;background:#fff8fc"><strong id="source-preview-label">所选原版本</strong><audio id="source-preview-audio" controls preload="metadata" style="width:100%;margin-top:8px"></audio><small>原版本试听独立显示，不会覆盖右侧的新生成结果。</small></section>
  <button type="button" class="secondary" id="align-take">获取词级时间戳</button>
  <label>对齐语言<select id="align-language"><option>Chinese</option><option>English</option><option>Cantonese</option><option>Japanese</option><option>Korean</option><option>German</option><option>French</option><option>Spanish</option></select></label>
  <canvas id="edit-wave" width="900" height="130" aria-label="拖动选择音频区间"></canvas><small id="wave-hint">试听加载后，拖动波形选择补写区间；也可以点击对齐词。</small>
  <button type="button" class="secondary" id="select-all-audio">选择整段音频</button><div id="word-timeline" class="word-timeline"></div>`;
$('session-field').append(toolsPanel);
const importForm=document.createElement('form');
importForm.id='project-import'; importForm.className='project-import';
importForm.innerHTML='<label>导入工程 / 会话<input type="file" name="file" accept=".safetensors" required></label><button class="secondary">导入工程</button>';
$('history-panel').prepend(importForm);
const projectLink=document.createElement('a');projectLink.id='project-download';projectLink.className='ghost';projectLink.download=true;projectLink.textContent='导出工程';$('output-links').append(projectLink);
const subtitleLink=document.createElement('a');subtitleLink.id='subtitle-download';subtitleLink.className='ghost';subtitleLink.download=true;subtitleLink.textContent='实际时间轴 SRT';subtitleLink.hidden=true;$('output-links').append(subtitleLink);
$('stream-mode').add(new Option('SRT 台词批量生成，导出实际时间轴','srt'));
const chunkField=document.createElement('label');chunkField.innerHTML='长文每段预算（UTF-8 字节）<input id="chunk-bytes" type="number" min="32" max="3000" value="300">';$('stream-fields').append(chunkField);
const srtFile=document.createElement('label');srtFile.innerHTML='导入 SRT 文件<input id="srt-file" type="file" accept=".srt,text/plain">';$('stream-fields').append(srtFile);$('srt-file').onchange=async e=>{const input=e.target,file=input.files[0];if(!file)return;try{const text=await file.text();if(input.files[0]!==file)return;$('text').value=text;$('stream-mode').value='srt';}catch(err){if(input.files[0]===file)notice(err.message)}};
const settingsForm=document.createElement('form');settingsForm.className='runtime-form';
const modelLabels={model:'DRAMA 主模型',codec:'音频 Codec',text_head:'完整文本头（兼容回退）',line_head:'小场景控制头（可从完整头读取）',asr_model:'Qwen3-ASR 模型',aligner:'词级对齐模型（目录或 safetensors）',r2t2_model:'Confucius Q8 主模型',r2t2_projector:'Confucius Q8 音频投影器',r2t2_vad:'FireRedVAD 模型目录'};
settingsForm.innerHTML=`<p class="hint full-width">可填写任意磁盘的绝对路径（如 E:\\Models\\模型.gguf），也支持网络共享路径。相对路径以本程序的 models 目录为起点；不同模型可以分别存放。保存只需主模型、Codec 和所选场景头；转写、对齐与 VAD 可留空，使用对应功能时才检查文件。</p>
  ${Object.entries(modelLabels).map(([key,label])=>`<label>${label}<input id="runtime-${key}" type="text" spellcheck="false" ${["model","codec"].includes(key)?"required":""}><small class="model-path-state" id="path-${key}"></small></label>`).join('')}
  <label>音频转写引擎<select id="runtime-asr"><option value="r2t2">Confucius R2T2 · Q8 原生</option><option value="qwen">Qwen3-ASR · 原生 Torch</option></select></label>
  <label>线性注意力<select id="runtime-gdn"><option value="torch">Torch · 通用模式</option><option value="fla">FLA · Triton 融合算子</option></select></label>
  <label>场景控制头<select id="runtime-scene-head"><option value="compact">精简控制头</option><option value="full">完整文本头 · 兼容模式</option></select></label>
  <label>专家驻留<select id="runtime-placement"><option value="ram">系统内存 RAM · 显存缓存（默认）</option><option value="vram">单卡显存 VRAM · 全部驻留</option></select><small>VRAM 需容纳完整主模型、专家、场景头和运行上下文；不足会明确拒绝，不自动改为内存流式。实际需求按所选模型检查。</small></label>
  <label>专家计算<select id="runtime-moe"><option value="auto">自动选择</option><option value="cuda">CUDA 专家算子</option><option value="torch">通用 Torch</option></select></label>
  <label><input id="runtime-cuda-graph" type="checkbox" checked>启用可用的 CUDA Graph</label>
  <label>专家显存缓存（GiB，0 为自动）<input id="runtime-cache" type="number" min="0" max="128" step="0.5" value="12"></label>
  <button class="primary">保存并应用运行设置</button><p id="acceleration-detail" class="hint"></p>`;
$('environment-panel').append(settingsForm);
function updatePlacementControls(){$('runtime-cache').disabled=$('runtime-placement').value==='vram';}
$('runtime-placement').onchange=updatePlacementControls;
let optionsLoaded=false, observedAlignment=null, alignmentGeneration=0, loadedSession='',takeSession='', waveBuffer=null, dragStart=null;
let observedPreview=null, previewGeneration=0, waveGeneration=0, selectedSessionInfo=null, editRangeContext='', selectedDuration=Infinity;
let selectedSourcePreview=null,previewSubmission=null,sourcePreparation=null;
window.getEditorSourcePreview=()=>selectedSourcePreview;
for(const id of ['start-seconds','end-seconds'])$(id).step='any';
function loadReferenceSpeakers(){try{const saved=JSON.parse(localStorage.getItem('drama-reference-speakers')||'{}');if(saved&&typeof saved==='object'&&!Array.isArray(saved))return Object.fromEntries(Object.entries(saved).filter(([id,value])=>/^[0-9a-f]{32}$/.test(id)&&Number.isInteger(value)&&value>=0&&value<16));}catch{}return {};}
let referenceSpeakers=loadReferenceSpeakers();
window.getReferenceSpeakers=()=>referenceSpeakers;
function editDuration(){return Number.isFinite(selectedDuration)?selectedDuration:waveBuffer?.duration||Infinity;}
function timeField(value){return value.toFixed(Math.abs(value/.05-Math.round(value/.05))<1e-7?3:6);}
function selectInterval(start,end){try{const duration=editDuration(),range=window.DramaTimeSelection.nearestRange(start,end,duration);$('start-seconds').value=timeField(range.start);$('end-seconds').value=timeField(range.end);let info=$('edit-selection-info');if(!info){info=document.createElement('p');info.id='edit-selection-info';info.className='hint';$('wave-hint').after(info);}info.textContent=`选区 ${range.start.toFixed(3)}–${range.end.toFixed(3)} 秒 · ${(range.end-range.start).toFixed(3)} 秒 · ${range.start===0&&Math.abs(range.end-duration)<1e-6?'整段音频':'已自动对齐最近有效时间'}；新片段长度默认由模型决定。`;drawWave();return range;}catch(error){notice(error.message);throw error;}}
$('select-all-audio').onclick=()=>{if(Number.isFinite(editDuration()))selectInterval(0,editDuration());else notice('请先选择有音频的会话。');};
window.snapCurrentEditRange=()=>selectInterval(Number($('start-seconds').value),Number($('end-seconds').value));
function drawWave(){const c=$('edit-wave'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#fff4f8';ctx.fillRect(0,0,c.width,c.height);if(!waveBuffer)return;
 const data=waveBuffer.getChannelData(0),stride=Math.max(1,Math.floor(data.length/c.width));ctx.strokeStyle='#ee5e9e';ctx.beginPath();
 for(let x=0;x<c.width;x++){let lo=0,hi=0;for(let i=x*stride;i<Math.min((x+1)*stride,data.length);i++){lo=Math.min(lo,data[i]);hi=Math.max(hi,data[i]);}ctx.moveTo(x,65+lo*55);ctx.lineTo(x,65+hi*55);}ctx.stroke();
 const a=Number($('start-seconds').value)/waveBuffer.duration*c.width,b=Number($('end-seconds').value)/waveBuffer.duration*c.width;ctx.fillStyle='#f184b455';ctx.fillRect(a,0,b-a,c.height);
}
for(const id of ['audio','source-preview-audio'])$(id).addEventListener('loadedmetadata',async()=>{const audio=$(id),source=audio.src,generation=waveGeneration;if(!audio.getAttribute('src')||id==='audio'&&!$('source-preview').hidden)return;let context;try{context=new AudioContext();const buffer=await context.decodeAudioData(await(await fetch(source)).arrayBuffer());if(audio.src===source&&generation===waveGeneration&&(id==='source-preview-audio'||$('source-preview').hidden)){waveBuffer=buffer;drawWave();}}catch(e){if(generation===waveGeneration)notice(e.message)}finally{await context?.close();}});
$('source-preview-audio').onplay=()=>$('audio').pause();$('audio').onplay=()=>$('source-preview-audio').pause();
$('edit-wave').onpointerdown=e=>{if(!waveBuffer)return;dragStart=Math.max(0,Math.min(1,(e.clientX-e.target.getBoundingClientRect().left)/e.target.getBoundingClientRect().width))*waveBuffer.duration;e.target.setPointerCapture(e.pointerId);};
$('edit-wave').onpointermove=e=>{if(dragStart===null||!waveBuffer)return;const end=Math.max(0,Math.min(1,(e.clientX-e.target.getBoundingClientRect().left)/e.target.getBoundingClientRect().width))*waveBuffer.duration;selectInterval(Math.min(dragStart,end),Math.max(dragStart,end));};
$('edit-wave').onpointerup=()=>{dragStart=null};
for(const id of ['start-seconds','end-seconds'])$(id).addEventListener('input',drawWave);
for(const id of ['start-seconds','end-seconds'])$(id).addEventListener('change',()=>{try{$(id).value=timeField(window.DramaTimeSelection.nearestValue(Number($(id).value),editDuration()));drawWave();}catch(error){notice(error.message);}});
async function selectedSession(){if(!$('session').value)throw Error('请先选择会话');return $('session').value;}
function clearSourcePreview(){previewGeneration++;waveGeneration++;observedPreview=null;selectedSourcePreview=null;window.invalidateInfillSource?.();waveBuffer=null;$('source-preview-audio').pause();$('source-preview-audio').removeAttribute('src');$('source-preview-audio').load();$('source-preview').hidden=true;$('word-timeline').replaceChildren();drawWave();}
function clearEditorPreview(){window.dismissAudioPreview?.();window.stopLivePreview?.();$('output-links').hidden=true;clearSourcePreview();$('audio').pause();$('audio').removeAttribute('src');$('audio').load();}
window.clearEditorPreview=clearEditorPreview;
function submitTool(mode){
 if(mode!=='preview')return requestTool(mode);
 const sessionId=$('session').value,takeIndex=Number($('take-index').value);
 if(previewSubmission?.sessionId===sessionId&&previewSubmission.takeIndex===takeIndex)return previewSubmission.promise;
 if(observedPreview?.sessionId===sessionId&&observedPreview.takeIndex===takeIndex)return Promise.resolve();
 const pending={sessionId,takeIndex};previewSubmission=pending;
 pending.promise=requestTool(mode).finally(()=>{if(previewSubmission===pending)previewSubmission=null;});
 return pending.promise;
}
async function requestTool(mode){
 let generation;
 const current=()=>generation===undefined||generation===(mode==='align'?alignmentGeneration:previewGeneration);
 try{
  const id=await selectedSession(),index=Number($('take-index').value);
  if(mode==='align'){generation=++alignmentGeneration;observedAlignment=null;}
  else{generation=++previewGeneration;observedPreview=null;}
  const submitted=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,session_id:id,take_index:index,language:$('align-language').value})});
  if(!current())return;
  if(mode==='preview'&&$('session').value===id&&Number($('take-index').value)===index)observedPreview={jobId:submitted.job_id,sessionId:id,takeIndex:index,generation};
  if(mode==='align')observedAlignment={jobId:submitted.job_id,sessionId:id,takeIndex:index};
  notice(mode==='align'?'词级对齐已加入任务队列。':'试听解码已加入任务队列。');
 }catch(e){if(current())notice(e.message)}
}
window.prepareInfillSource=()=>{
 if(window.hasCurrentInfillSource?.())return Promise.resolve();
 if(sourcePreparation)return sourcePreparation;
 const sessionId=$('session').value,takeIndex=Number($('take-index').value);
 sourcePreparation=(async()=>{
  if(!sessionId)throw Error('请先选择含音频的会话。');
  if(selectedSourcePreview)window.bindInfillPlaybackSource?.(selectedSourcePreview,'source-preview-audio');
  if(window.hasCurrentInfillSource?.())return;
  notice('正在自动准备所选来源音频…');
  await submitTool('preview');
  const deadline=Date.now()+120000;
  while(Date.now()<deadline){
   if($('session').value!==sessionId||Number($('take-index').value)!==takeIndex)throw Error('来源会话或版本已改变，请确认后重新生成。');
   await updateTools();
   if(window.hasCurrentInfillSource?.())return;
   if(!observedPreview)throw Error('所选来源音频未能准备完成，请查看试听任务中的原因。');
   await new Promise(resolve=>setTimeout(resolve,250));
  }
  throw Error('来源音频仍在准备，请稍候再生成。');
 })().finally(()=>{sourcePreparation=null;});
 return sourcePreparation;
};
function invalidateAlignment(){alignmentGeneration++;observedAlignment=null;}
$('align-take').onclick=()=>submitTool('align');$('preview-take').onclick=()=>submitTool('preview');
$('session').addEventListener('change',async()=>{invalidateAlignment();clearSourcePreview();loadedSession='';editRangeContext='';await updateTools();if($('session').value)void submitTool('preview');});
$('take-index').addEventListener('change',async()=>{invalidateAlignment();clearSourcePreview();editRangeContext='';await updateTools();void submitTool('preview');});

let importGeneration=0,settingsGeneration=0,settingsJob=null;
importForm.onsubmit=async e=>{e.preventDefault();const generation=++importGeneration;try{const r=await api('/api/projects/import',{method:'POST',body:new FormData(importForm)});if(generation!==importGeneration)return;await refreshStatus();if(generation!==importGeneration)return;clearEditorPreview();$('session').value=r.session_id;showPage('infill');loadedSession='';await updateTools();if(generation===importGeneration)notice('工程已导入，参考与演出历史已恢复。');}catch(err){if(generation===importGeneration)notice(err.message)}};
settingsForm.onsubmit=async e=>{e.preventDefault();const generation=++settingsGeneration;settingsJob=null;try{const settings=Object.fromEntries(Object.keys(modelLabels).map(key=>[key,$(`runtime-${key}`).value]));Object.assign(settings,{asr_backend:$('runtime-asr').value,gdn_backend:$('runtime-gdn').value,scene_head_mode:$('runtime-scene-head').value,moe_backend:$('runtime-moe').value,expert_placement:$('runtime-placement').value,cuda_graph:$('runtime-cuda-graph').checked,expert_cache_gb:Number($('runtime-cache').value)});const submitted=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'configure',settings})});if(generation!==settingsGeneration)return;settingsJob=submitted.job_id;notice('运行设置已加入队列，应用时释放已加载模型。');}catch(err){if(generation===settingsGeneration)notice(err.message)}};
const originalShowResult=showResult;showResult=function(job){if(job.result?.audio){waveBuffer=null;drawWave();}originalShowResult(job);showResultIdentity(job);if(!job.result?.audio)return;clearSourcePreview();if(job.result.metrics?.duration_seconds>0){selectedDuration=job.result.metrics.duration_seconds;selectInterval(0,selectedDuration);}if(takeSession!==job.result.session_id){$('take-index').replaceChildren(new Option('最新版本',-1));takeSession=job.result.session_id;}$('word-timeline').replaceChildren();$('project-download').href=job.result.project||'';$('project-download').hidden=!job.result.project;$('subtitle-download').hidden=!job.result.subtitle;if(job.result.subtitle)$('subtitle-download').href=job.result.subtitle;if(job.result.metrics?.length_limit_reached)notice('本次生成达到最大音频帧限制，内容可能未完整；参数记录已标记截断。');if(job.result.metrics?.acceleration_fallback)notice(`本次加速已回退 Torch：${job.result.metrics.acceleration_fallback}`);loadedSession='';};
async function updateTools(){try{const s=await api('/api/status');if(!optionsLoaded){for(const key of Object.keys(modelLabels))$(`runtime-${key}`).value=s.settings[key]||'';$('runtime-asr').value=s.settings.asr_backend;$('runtime-gdn').value=s.settings.gdn_backend;$('runtime-scene-head').value=s.settings.scene_head_mode||'compact';$('runtime-cache').value=s.settings.expert_cache_gb;$('runtime-moe').value=s.settings.moe_backend||'auto';$('runtime-placement').value=s.settings.expert_placement||'ram';updatePlacementControls();$('runtime-cuda-graph').checked=s.settings.cuda_graph!==false;optionsLoaded=true;}
 for(const [key,state] of Object.entries(s.model_paths||{})){const el=$(`path-${key}`);el.textContent=`${state.ready?'已找到':'未找到'} · ${state.path}`;el.classList.toggle('missing',!state.ready);}
 if(settingsJob){const observed=settingsJob,job=await observedJob(observed,await api('/api/jobs'));if(settingsJob===observed&&job&&!['queued','running'].includes(job.status)){notice(job.status==='done'?'模型路径与转写设置已保存，下次启动会继续使用。':job.error||'设置未应用');settingsJob=null;}}
 $('acceleration-detail').textContent=describeAcceleration(s.acceleration);
 const released=describeUnloadMemory(s.last_unload_memory);$('release-memory-detail').textContent=released;$('release-memory-detail').hidden=!released;
 for(const input of document.querySelectorAll('.reference-choice')){if(input.parentNode.querySelector('select'))continue;const select=document.createElement('select');select.className='speaker-select';select.setAttribute('aria-label','角色编号');select.add(new Option('原稿角色',''));for(let i=0;i<16;i++)select.add(new Option(`角色 ${i}`,i));select.value=referenceSpeakers[input.value]??'';select.onchange=()=>{if(select.value==='')delete referenceSpeakers[input.value];else referenceSpeakers[input.value]=Number(select.value);try{localStorage.setItem('drama-reference-speakers',JSON.stringify(referenceSpeakers));}catch{notice('角色映射已临时应用；浏览器无法保存，刷新后请重新选择。');}};input.parentNode.append(select);}
 const id=$('session').value;if(id&&id!==loadedSession){const session=await api(`/api/sessions/${id}`);if($('session').value===id){const selected=takeSession===id?$('take-index').value:'-1';$('take-index').replaceChildren(new Option('最新版本',-1));session.takes.forEach((t,i)=>$('take-index').add(new Option(`第 ${i+1} 遍 · ${t.text.slice(0,25)}`,i)));$('take-index').value=[...$('take-index').options].some(option=>option.value===selected)?selected:'-1';loadedSession=id;takeSession=id;selectedSessionInfo=session;$('direction').value=session.direction||'';}}
 if(selectedSessionInfo?.session_id===id&&$('session').value===id){const key=id+':'+$('take-index').value,index=Number($('take-index').value),take=selectedSessionInfo.takes[index<0?selectedSessionInfo.takes.length-1:index];if(take&&key!==editRangeContext){selectedDuration=take.duration;editRangeContext=key;if(selectedDuration>0)selectInterval(0,selectedDuration);}}
 const preview=observedPreview;if(preview){const j=await observedJob(preview.jobId,await api('/api/jobs'));if(preview===observedPreview&&preview.generation===previewGeneration&&j&&!['queued','running'].includes(j.status)){observedPreview=null;if(j.status==='done'&&$('session').value===preview.sessionId&&Number($('take-index').value)===preview.takeIndex){selectedSourcePreview=j.result;$('source-preview-label').textContent='所选来源 · 第 '+(j.result.take_index+1)+' 遍 · '+(j.result.metrics?.duration_seconds||0).toFixed(2)+' 秒';$('source-preview-audio').src=j.result.audio;$('source-preview').hidden=false;window.bindInfillPlaybackSource?.(j.result,'source-preview-audio');}else if(j.status!=='done')notice(j.error||'来源试听未完成');}}
 const pending=observedAlignment;
 if(pending){const j=await observedJob(pending.jobId,await api('/api/jobs'));if(pending===observedAlignment&&j&&!['queued','running'].includes(j.status)){observedAlignment=null;if(j.status==='done'){if($('session').value===pending.sessionId&&Number($('take-index').value)===pending.takeIndex){$('word-timeline').replaceChildren();for(const w of j.result.alignment){const button=document.createElement('button');button.type='button';button.className='word';button.textContent=w.text;button.title=`${w.start.toFixed(2)}–${w.end.toFixed(2)} 秒`;button.onclick=()=>selectInterval(w.start,w.end);$('word-timeline').append(button);}notice('对齐完成，点击词选择区间；拖动波形可选多个词。');}else notice('对齐已完成；切回原会话与版本后可重新加载词边界。');}else notice(j.error||'对齐已取消');}}

 }catch(e){notice(e.message)}}
async function pollTools(){await updateTools();setTimeout(pollTools,2100)}pollTools();

function describeAcceleration(a){
 const moe=a.moe_backend&&a.moe_backend!=='not_run'?`上次专家计算：${a.moe_backend}`:'专家计算：尚未运行';
 const graph=a.graph_stats?`CUDA Graph：${a.graph_stats.captures||0} 层捕获 / ${a.graph_stats.replays||0} 次复用，音频塔 ${a.graph_stats.tower_replays||0} 次复用`:'CUDA Graph：尚无实测';
 const reasons=[a.moe_fallback,a.graph_stats?.disabled_reason,...(a.graph_stats?.fallback_reasons||[])].filter(Boolean);
 const placement=a.expert_placement==='vram'?'专家驻留：单卡显存':'专家驻留：系统内存 / 显存缓存';
 return `${placement}；${moe}；${graph}；线性注意力可选 ${a.fla_installed?'Torch / FLA':'Torch'}。${reasons.length?'回退说明：'+[...new Set(reasons)].join('；'):''}`;
}

function describeUnloadMemory(report){
 if(!report||typeof report!=='object')return '';
 if(report.status==='refused_active')return '上次释放未完成：仍有任务占用，模型保持加载。';
 if(report.status!=='ok')return '上次释放未完整结束，请查看任务记录中的原因。';
 const before=report.before||{},after=report.after||{};
 const gib=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0?(value/1024**3).toFixed(1):null;
 const pair=(first,last)=>{const a=gib(first),b=gib(last);return a!==null&&b!==null?`${a} → ${b} GiB`:b!==null?`${b} GiB`:'暂无统计';};
 const pinned=phase=>phase.host_allocator?.status==='measured'?phase.host_allocator['allocated_bytes.current']:undefined;
 const parts=[`固定内存 ${pair(pinned(before),pinned(after))}`,`进程内存 ${pair(before.process?.rss,after.process?.rss)}`];
 const known=Object.values(after.owned_objects||{}).filter(value=>value?.weakref_supported===true&&typeof value.alive==='boolean');
 if(known.length){const alive=known.filter(value=>value.alive).length;parts.push(alive?`仍有 ${alive} 个可核对的模型对象引用`:`可核对模型对象 ${known.length}/${known.length} 已清理`);}
 return '最近释放：'+parts.join('；')+'。进程内存还包含运行库和其他缓存。';
}

// One shared policy across the basic generation tabs.
const deliveryChoice=$('replace-delivery-tags').closest('label');
deliveryChoice.lastChild.textContent='使用新要求时，覆盖原有情绪／语速标签';
$('generate-form').insertBefore(deliveryChoice,$('text-label'));
$('replace-delivery-tags').disabled=false;
function markPreviousResult(message){const box=$('result-identity');if(!box?.dataset.jobId)return;let note=$('result-stale-note');if(!note){note=document.createElement('p');note.id='result-stale-note';note.style.cssText='color:#b26083;font-size:12px';box.after(note);}note.textContent=message;note.hidden=false;}
$('generate-form').addEventListener('input',()=>markPreviousResult('输入已修改，右侧仍是上一次结果；点击生成后才会使用新设置。'));
$('generate-form').addEventListener('change',event=>markPreviousResult(['session','take-index'].includes(event.target.id)?'左侧来源已改变；右侧保留刚才的生成结果，便于对比。':'设置已修改，右侧仍是上一次结果；点击生成后使用新设置。'));
$('tabs').addEventListener('click',()=>{const mode={generate:'tts',scene:'scene',revise:'revise',infill:'infill',stream:$('stream-mode').value}[page];if(mode&&$('result-identity')?.dataset.mode!==mode)markPreviousResult('右侧是上一次其他功能的结果，当前页面的新设置尚未生成。');$('direction').disabled=page==='infill';});
const identifiedShowResult=showResult;showResult=function(job){identifiedShowResult(job);if($('result-stale-note'))$('result-stale-note').hidden=true;};

const castHelp=document.createElement('div');castHelp.className='hint';castHelp.style.cssText='margin:8px 0 14px';
const castButtons=document.createElement('div'),castExplain=document.createElement('p');castExplain.textContent='人物描述要与台词角色对应：每行写“导演：台词”或“女主：台词”。下方可插入角色，修改设置后需重新生成。';castHelp.append(castExplain,castButtons);$('characters').closest('label').after(castHelp);
function sceneRoles(){return [...new Set($('characters').value.split(/\r?\n/).map(line=>line.match(/^\s*([^:：<>\[\]]{1,48})[:：]/)?.[1]?.trim()).filter(Boolean))];}
function refreshSceneRoles(){castButtons.replaceChildren();for(const name of sceneRoles()){const b=document.createElement('button');b.type='button';b.className='secondary';b.textContent='插入 '+name;b.onclick=()=>{const t=$('text'),at=t.selectionStart,start=t.value.lastIndexOf('\n',Math.max(0,at-1))+1,line=t.value.slice(start),old=sceneRoles().find(role=>line.startsWith(role+':')||line.startsWith(role+'：')),remove=old?old.length+1:0,prefix=name+'：';t.setRangeText(prefix,start,start+remove,'end');t.focus();t.dispatchEvent(new Event('input',{bubbles:true}));};castButtons.append(b);}}
$('characters').addEventListener('input',refreshSceneRoles);refreshSceneRoles();
