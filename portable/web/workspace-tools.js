const toolsPanel = document.createElement('div');
toolsPanel.className = 'take-tools';
toolsPanel.innerHTML = `<label>会话中的演出版本<select id="take-index"><option value="-1">最新版本</option></select></label>
  <button type="button" class="secondary" id="preview-take">试听所选版本</button>
  <button type="button" class="secondary" id="align-take">获取词级时间戳</button>
  <label>对齐语言<select id="align-language"><option>Chinese</option><option>English</option><option>Cantonese</option><option>Japanese</option><option>Korean</option><option>German</option><option>French</option><option>Spanish</option></select></label>
  <canvas id="edit-wave" width="900" height="130" aria-label="拖动选择音频区间"></canvas><small id="wave-hint">试听加载后，拖动波形选择补写区间；也可以点击对齐词。</small>
  <div id="word-timeline" class="word-timeline"></div>`;
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
const modelLabels={model:'DRAMA 主模型',codec:'音频 Codec',text_head:'场景完整文本头',asr_model:'Qwen3-ASR 模型',aligner:'词级对齐模型目录',r2t2_model:'Confucius Q8 主模型',r2t2_projector:'Confucius Q8 音频投影器',r2t2_vad:'FireRedVAD 模型目录'};
settingsForm.innerHTML=`<p class="hint full-width">可填写任意磁盘的绝对路径（如 E:\\Models\\模型.gguf），也支持网络共享路径。相对路径以本程序的 models 目录为起点；不同模型可以分别存放。</p>
  ${Object.entries(modelLabels).map(([key,label])=>`<label>${label}<input id="runtime-${key}" type="text" spellcheck="false" required><small class="model-path-state" id="path-${key}"></small></label>`).join('')}
  <label>音频转写引擎<select id="runtime-asr"><option value="r2t2">Confucius R2T2 · Q8 原生</option><option value="qwen">Qwen3-ASR · 原生 Torch</option></select></label>
  <label>线性注意力<select id="runtime-gdn"><option value="torch">Torch · 通用模式</option><option value="fla">FLA · Triton 融合算子</option></select></label>
  <label>专家显存缓存（GiB，0 为自动）<input id="runtime-cache" type="number" min="0" max="128" step="0.5" value="12"></label>
  <button class="primary">保存并应用运行设置</button><p id="acceleration-detail" class="hint"></p>`;
$('environment-panel').append(settingsForm);
let optionsLoaded=false, observedAlignment=null, loadedSession='',takeSession='', waveBuffer=null, dragStart=null;
function loadReferenceSpeakers(){try{const saved=JSON.parse(localStorage.getItem('drama-reference-speakers')||'{}');if(saved&&typeof saved==='object'&&!Array.isArray(saved))return Object.fromEntries(Object.entries(saved).filter(([id,value])=>/^[0-9a-f]{32}$/.test(id)&&Number.isInteger(value)&&value>=0&&value<16));}catch{}return {};}
let referenceSpeakers=loadReferenceSpeakers();
window.getReferenceSpeakers=()=>referenceSpeakers;
function selectInterval(start,end){$('start-seconds').value=start.toFixed(3);$('end-seconds').value=end.toFixed(3);drawWave();}
function drawWave(){const c=$('edit-wave'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#fff4f8';ctx.fillRect(0,0,c.width,c.height);if(!waveBuffer)return;
 const data=waveBuffer.getChannelData(0),stride=Math.max(1,Math.floor(data.length/c.width));ctx.strokeStyle='#ee5e9e';ctx.beginPath();
 for(let x=0;x<c.width;x++){let lo=0,hi=0;for(let i=x*stride;i<Math.min((x+1)*stride,data.length);i++){lo=Math.min(lo,data[i]);hi=Math.max(hi,data[i]);}ctx.moveTo(x,65+lo*55);ctx.lineTo(x,65+hi*55);}ctx.stroke();
 const a=Number($('start-seconds').value)/waveBuffer.duration*c.width,b=Number($('end-seconds').value)/waveBuffer.duration*c.width;ctx.fillStyle='#f184b455';ctx.fillRect(a,0,b-a,c.height);
}
$('audio').addEventListener('loadedmetadata',async()=>{const source=$('audio').src;let context;try{context=new AudioContext();const buffer=await context.decodeAudioData(await(await fetch(source)).arrayBuffer());if($('audio').src===source){waveBuffer=buffer;drawWave();}}catch(e){notice(e.message)}finally{await context?.close();}});
$('edit-wave').onpointerdown=e=>{if(!waveBuffer)return;dragStart=Math.max(0,Math.min(1,(e.clientX-e.target.getBoundingClientRect().left)/e.target.getBoundingClientRect().width))*waveBuffer.duration;e.target.setPointerCapture(e.pointerId);};
$('edit-wave').onpointermove=e=>{if(dragStart===null||!waveBuffer)return;const end=Math.max(0,Math.min(1,(e.clientX-e.target.getBoundingClientRect().left)/e.target.getBoundingClientRect().width))*waveBuffer.duration;selectInterval(Math.min(dragStart,end),Math.max(dragStart,end));};
$('edit-wave').onpointerup=()=>{dragStart=null};
for(const id of ['start-seconds','end-seconds'])$(id).addEventListener('input',drawWave);
async function selectedSession(){if(!$('session').value)throw Error('请先选择会话');return $('session').value;}
function clearEditorPreview(){window.dismissAudioPreview?.();window.stopLivePreview?.();$('output-links').hidden=true;waveBuffer=null;$('audio').pause();$('audio').removeAttribute('src');$('word-timeline').replaceChildren();drawWave();}
window.clearEditorPreview=clearEditorPreview;
async function submitTool(mode){try{const id=await selectedSession(),index=Number($('take-index').value);const submitted=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,session_id:id,take_index:index,language:$('align-language').value})});if(mode==='preview'&&$('session').value===id&&Number($('take-index').value)===index)window.watchAudioJob(submitted.job_id,{sessionId:id,takeIndex:index});if(mode==='align')observedAlignment={jobId:submitted.job_id,sessionId:id,takeIndex:index};notice(mode==='align'?'词级对齐已加入任务队列。':'试听解码已加入任务队列。');}catch(e){notice(e.message)}}
$('align-take').onclick=()=>submitTool('align');$('preview-take').onclick=()=>submitTool('preview');
$('session').addEventListener('change',async()=>{clearEditorPreview();loadedSession='';await updateTools()});
$('take-index').addEventListener('change',clearEditorPreview);

let importGeneration=0,settingsGeneration=0,settingsJob=null;
importForm.onsubmit=async e=>{e.preventDefault();const generation=++importGeneration;try{const r=await api('/api/projects/import',{method:'POST',body:new FormData(importForm)});if(generation!==importGeneration)return;await refreshStatus();if(generation!==importGeneration)return;clearEditorPreview();$('session').value=r.session_id;showPage('infill');loadedSession='';await updateTools();if(generation===importGeneration)notice('工程已导入，参考与演出历史已恢复。');}catch(err){if(generation===importGeneration)notice(err.message)}};
settingsForm.onsubmit=async e=>{e.preventDefault();const generation=++settingsGeneration;settingsJob=null;try{const settings=Object.fromEntries(Object.keys(modelLabels).map(key=>[key,$(`runtime-${key}`).value]));Object.assign(settings,{asr_backend:$('runtime-asr').value,gdn_backend:$('runtime-gdn').value,expert_cache_gb:Number($('runtime-cache').value)});const submitted=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'configure',settings})});if(generation!==settingsGeneration)return;settingsJob=submitted.job_id;notice('运行设置已加入队列，应用时释放已加载模型。');}catch(err){if(generation===settingsGeneration)notice(err.message)}};
const originalShowResult=showResult;showResult=function(job){if(job.result?.audio){waveBuffer=null;drawWave();}originalShowResult(job);if(!job.result?.audio)return;if(takeSession!==job.result.session_id){$('take-index').replaceChildren(new Option('最新版本',-1));takeSession=job.result.session_id;}$('word-timeline').replaceChildren();$('project-download').href=job.result.project||'';$('project-download').hidden=!job.result.project;$('subtitle-download').hidden=!job.result.subtitle;if(job.result.subtitle)$('subtitle-download').href=job.result.subtitle;if(job.result.metrics?.length_limit_reached)notice('本次生成达到最大音频帧限制，内容可能未完整；参数记录已标记截断。');if(job.result.metrics?.acceleration_fallback)notice(`本次加速已回退 Torch：${job.result.metrics.acceleration_fallback}`);loadedSession='';};
async function updateTools(){try{const s=await api('/api/status');if(!optionsLoaded){for(const key of Object.keys(modelLabels))$(`runtime-${key}`).value=s.settings[key]||'';$('runtime-asr').value=s.settings.asr_backend;$('runtime-gdn').value=s.settings.gdn_backend;$('runtime-cache').value=s.settings.expert_cache_gb;optionsLoaded=true;}
 for(const [key,state] of Object.entries(s.model_paths||{})){const el=$(`path-${key}`);el.textContent=`${state.ready?'已找到':'未找到'} · ${state.path}`;el.classList.toggle('missing',!state.ready);}
 if(settingsJob){const observed=settingsJob,job=await observedJob(observed,await api('/api/jobs'));if(settingsJob===observed&&job&&!['queued','running'].includes(job.status)){notice(job.status==='done'?'模型路径与转写设置已保存，下次启动会继续使用。':job.error||'设置未应用');settingsJob=null;}}
 $('acceleration-detail').textContent=`全注意力：${s.acceleration.attention}；FLA：${s.acceleration.fla_installed?'已安装，可选择':'未安装'}；FlashAttention：${s.acceleration.flash_attn_installed?'算子已安装，当前模型未启用':'未安装'}；CUDA Graph：当前未启用。`;
 for(const input of document.querySelectorAll('.reference-choice')){if(input.parentNode.querySelector('select'))continue;const select=document.createElement('select');select.className='speaker-select';select.setAttribute('aria-label','角色编号');select.add(new Option('原稿角色',''));for(let i=0;i<16;i++)select.add(new Option(`角色 ${i}`,i));select.value=referenceSpeakers[input.value]??'';select.onchange=()=>{if(select.value==='')delete referenceSpeakers[input.value];else referenceSpeakers[input.value]=Number(select.value);try{localStorage.setItem('drama-reference-speakers',JSON.stringify(referenceSpeakers));}catch{notice('角色映射已临时应用；浏览器无法保存，刷新后请重新选择。');}};input.parentNode.append(select);}
 const id=$('session').value;if(id&&id!==loadedSession){const session=await api(`/api/sessions/${id}`);if($('session').value===id){const selected=takeSession===id?$('take-index').value:'-1';$('take-index').replaceChildren(new Option('最新版本',-1));session.takes.forEach((t,i)=>$('take-index').add(new Option(`第 ${i+1} 遍 · ${t.text.slice(0,25)}`,i)));$('take-index').value=[...$('take-index').options].some(option=>option.value===selected)?selected:'-1';loadedSession=id;takeSession=id;}}
 const pending=observedAlignment;
 if(pending){const j=await observedJob(pending.jobId,await api('/api/jobs'));if(pending===observedAlignment&&j&&!['queued','running'].includes(j.status)){observedAlignment=null;if(j.status==='done'){if($('session').value===pending.sessionId&&Number($('take-index').value)===pending.takeIndex){$('word-timeline').replaceChildren();for(const w of j.result.alignment){const button=document.createElement('button');button.type='button';button.className='word';button.textContent=w.text;button.title=`${w.start.toFixed(2)}–${w.end.toFixed(2)} 秒`;button.onclick=()=>selectInterval(w.start,w.end);$('word-timeline').append(button);}notice('对齐完成，点击词选择区间；拖动波形可选多个词。');}else notice('对齐已完成；切回原会话与版本后可重新加载词边界。');}else notice(j.error||'对齐已取消');}}

 }catch(e){notice(e.message)}}
async function pollTools(){await updateTools();setTimeout(pollTools,2100)}pollTools();
