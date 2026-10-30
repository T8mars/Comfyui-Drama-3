const asrTab=document.createElement('button');asrTab.type='button';asrTab.dataset.page='transcribe';asrTab.textContent='音频转写';$('tabs').insertBefore(asrTab,document.querySelector('[data-page="history"]'));
const asrPanel=document.createElement('section');asrPanel.id='transcribe-panel';asrPanel.className='card large-panel';asrPanel.hidden=true;
asrPanel.innerHTML=`<div class="section-title"><span class="icon">♫</span><div><h2>音频转写</h2><p>本地识别音频，检查文字后可用作角色参考稿件。</p></div></div>
 <div class="asr-grid"><form id="asr-form"><p id="asr-engine" class="hint"></p>
 <label class="drop">上传音频<input type="file" name="file" accept="audio/*" required><small>WAV / FLAC / OGG / MP3 / M4A，最大 512 MiB</small></label>
 <label>识别语言<select name="language"><option value="auto">自动检测</option></select></label>
 <label>热词<textarea name="hotwords" rows="3" placeholder="人名、地名、专业词汇，用逗号或换行分隔"></textarea></label>
 <label>上下文提示<textarea name="context" rows="2" placeholder="可选，简要说明音频主题"></textarea></label>
 <button class="primary">开始转写</button><button id="asr-settings" type="button" class="secondary">设置引擎与模型路径</button>
 </form><div><label>转写文字<textarea id="asr-text" rows="13" placeholder="结果将显示在这里，可校对后再使用"></textarea></label>
 <p id="asr-metrics" class="hint"></p><div class="output-links"><button id="asr-use" type="button" class="secondary">用作参考稿</button><a id="asr-download" class="ghost" hidden download>下载 TXT</a><a id="asr-metadata" class="ghost" hidden download>转写记录</a></div>
 <p class="hint">长音频使用 FireRedVAD 分段。热词用于引导识别；分段时间为近似值，精确词边界请使用词级对齐。</p></div></div>`;
$('environment-panel').before(asrPanel);
const showOriginalPage=showPage;showPage=function(next){showOriginalPage(next);asrPanel.hidden=next!=='transcribe';if(next==='transcribe')document.querySelector('.workspace').hidden=true;};
asrTab.onclick=()=>showPage('transcribe');$('asr-settings').onclick=()=>showPage('environment');
let observedASR=null,displayedASR=null,asrGeneration=0;
const asrLanguageLabels={auto:'自动检测',Chinese:'中文 / 中英混合',English:'英语',Cantonese:'粤语',
 Arabic:'阿拉伯语',German:'德语',French:'法语',Spanish:'西班牙语',Portuguese:'葡萄牙语',Indonesian:'印尼语',
 Italian:'意大利语',Korean:'韩语',Russian:'俄语',Thai:'泰语',Vietnamese:'越南语',Japanese:'日语',Turkish:'土耳其语',
 Hindi:'印地语',Malay:'马来语',Dutch:'荷兰语',Swedish:'瑞典语',Danish:'丹麦语',Finnish:'芬兰语',Polish:'波兰语',
 Czech:'捷克语',Filipino:'菲律宾语',Persian:'波斯语',Greek:'希腊语',Romanian:'罗马尼亚语',Hungarian:'匈牙利语',Macedonian:'马其顿语'};
let asrLanguageSignature='';
function updateASRLanguages(status){
 const backend=status.settings.asr_backend,capability=status.asr_languages;
 // Only the applied backend may advertise choices. During mixed-version
 // startup or missing metadata, auto remains valid for both implementations.
 const advertised=capability?.backend===backend&&Array.isArray(capability.languages)?capability.languages:[];
 const languages=['auto',...new Set(advertised.filter(name=>typeof name==='string'&&name&&name!=='auto'))];
 const signature=JSON.stringify([backend,languages]);
 if(signature===asrLanguageSignature)return;
 const select=$('asr-form').elements.language,previous=select.value;
 select.replaceChildren(...languages.map(name=>new Option(asrLanguageLabels[name]||name,name)));
 select.value=languages.includes(previous)?previous:'auto';
 asrLanguageSignature=signature;
}
window.showTranscription=function(job,submittedFile=null){
 asrGeneration++;observedASR=null;
 const r=job.result,id=job.payload?.input_id;
 const legacySource=/^[0-9a-f]{32}\.(wav|flac|ogg|mp3|m4a)$/.test(id||'')?`/api/files/input/${id}`:null;
 displayedASR={jobId:job.id,url:r.source_audio||legacySource,name:r.source_name||job.payload?.name||id||'reference.wav',file:submittedFile};
 $('asr-text').value=r.text||'';
 $('asr-metrics').textContent=`${r.backend==='r2t2'?'Confucius R2T2 Q8':'Qwen3-ASR'} · ${r.language||'未检测到语言'} · 音频 ${r.duration_seconds.toFixed(1)} 秒 · 转写 ${r.elapsed_seconds.toFixed(2)} 秒${r.truncated?' · 结果已截断，请检查':''}`;
 for(const [id,url] of [['asr-download',r.text_file],['asr-metadata',r.metadata]]){$(id).hidden=!url;if(url)$(id).href=url;else $(id).removeAttribute('href');}
 showPage('transcribe');
};
$('asr-form').onsubmit=async e=>{
 e.preventDefault();const button=e.target.querySelector('button[type=submit],button:not([type])');
 const data=new FormData(e.target),file=e.target.elements.file.files[0],generation=++asrGeneration;observedASR=null;button.disabled=true;
 try{const result=await api('/api/transcribe',{method:'POST',body:data});if(generation!==asrGeneration)return;observedASR={id:result.job_id,file};notice('音频转写已加入队列。');}
 catch(err){if(generation===asrGeneration)notice(err.message)}finally{button.disabled=false}
};
$('asr-use').onclick=async()=>{
 const text=$('asr-text').value,source=displayedASR;
 if(!text.trim()){notice('请先转写音频或填写参考稿。');return;}
 const button=$('asr-use');button.disabled=true;
 try{
  let file=source?.file;
  if(!file&&source?.url){const response=await fetch(source.url);if(!response.ok)throw Error('原始转写音频不存在，请重新上传并核对稿件。');file=new File([await response.blob()],source.name);}
  if(source!==displayedASR)throw Error('当前转写结果已改变，请重新选择参考稿。');
  if(!source)file=$('asr-form').elements.file.files[0];
  if(!file)throw Error('请先选择与稿件对应的参考音频。');
  const transfer=new DataTransfer();transfer.items.add(file);$('voice-form').elements.file.files=transfer.files;
  $('voice-form').elements.transcript.value=text;$('voice-form').closest('details').open=true;showPage('voices');notice('对应的原始音频与参考稿已填入，请核对后保存音色。');
 }catch(err){notice(err.message)}finally{button.disabled=false}
};
async function pollASR(){try{
 const s=await api('/api/status');updateASRLanguages(s);$('asr-engine').textContent=`当前引擎：${s.settings.asr_backend==='r2t2'?'Confucius R2T2 · Q8 原生':'Qwen3-ASR · 原生 Torch'}。在“环境与加速”中修改并保存模型路径。`;
 const pending=observedASR;if(pending){const job=await observedJob(pending.id,await api('/api/jobs'));if(pending===observedASR&&job&&!['queued','running'].includes(job.status)){observedASR=null;if(job.status==='done'){showTranscription(job,pending.file);notice(job.result.truncated?'转写达到长度限制，请检查结果。':'转写完成。');}else notice(job.error||'转写已取消');}}
 }catch(err){notice(err.message)}setTimeout(pollASR,1800)
}pollASR();
