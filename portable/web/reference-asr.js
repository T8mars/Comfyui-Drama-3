/* Local, editable reference transcripts. Never overwrite a newer file or edit. */
(() => {
"use strict";
const styles=document.createElement('style');styles.textContent=`.reference-asr-tools{padding:11px 13px;margin:10px 0 12px;border:1px solid #f0dce7;background:#fff9fc;border-radius:11px;font-family:Segoe UI,Microsoft YaHei UI,sans-serif}.reference-asr-tools .ref-asr-row{display:flex;flex-wrap:wrap;align-items:center;gap:10px}.reference-asr-tools label{display:inline-flex!important;align-items:center;gap:6px;margin:0;font-size:11px;color:#9a6b86}.reference-asr-tools input[type=checkbox]{width:15px;height:15px;margin:0;accent-color:#e85e96}.reference-asr-tools button{padding:7px 12px;border:1px solid #f0cfe0;border-radius:8px;background:white;color:#b15b87;font-size:11px;cursor:pointer}.reference-asr-tools p{font-size:11px;line-height:1.5;color:#a1849b;margin:7px 0 0}.reference-asr-tools p[data-error=true]{color:#bf526f}.reference-asr-tools details{margin-top:8px;font-size:11px;color:#a77994}.reference-asr-tools textarea{width:100%;margin:8px 0;min-height:66px;font:12px/1.6 Segoe UI,Microsoft YaHei UI,sans-serif}`;document.head.append(styles);
const bindings=[
 {input:document.querySelector('#voice-form input[type=file]'),target:document.querySelector('#voice-form textarea[name=transcript]')},
 {input:document.getElementById('cd-ref-file'),target:document.getElementById('cd-ref-text'),start:document.getElementById('cd-ref-start'),end:document.getElementById('cd-ref-end')}
];
async function json(url,options){const r=await fetch(url,options);let data;try{data=await r.json();}catch{throw Error('本地服务没有返回有效结果');}if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'请求未完成，请检查模型路径');return data;}
for(const b of bindings){
 if(!b.input||!b.target)continue;
 const tools=document.createElement('section');tools.className='reference-asr-tools';tools.setAttribute('aria-label','参考音频自动转写');
 const row=document.createElement('div');row.className='ref-asr-row';const label=document.createElement('label'),toggle=document.createElement('input');toggle.type='checkbox';toggle.defaultChecked=true;toggle.checked=true;label.append(toggle,document.createTextNode('上传后自动转写'));
 const button=document.createElement('button');button.type='button';button.textContent='重新转写';
 const status=document.createElement('p');status.setAttribute('role','status');status.textContent='选择音频后自动使用当前本地 ASR，结果可直接修改。';
 const detail=document.createElement('details'),summary=document.createElement('summary'),proposal=document.createElement('textarea'),adopt=document.createElement('button');detail.hidden=true;summary.textContent='查看转写结果';proposal.readOnly=true;proposal.setAttribute('aria-label','ASR 转写原稿');adopt.type='button';adopt.textContent='采用这份转写';detail.append(summary,proposal,adopt);row.append(label,button);tools.append(row,status,detail);b.target.closest('label').before(tools);
 b.target.placeholder='选择参考音频后自动转写。请试听、校稿，再编码保存。';
 const s={epoch:0,revision:0,writing:false,busy:false,job:null,timer:null};
 const source=()=>b.input.files?.[0]||window.referencePreviewFile?.(b.input)||null;
 const setStatus=(message,error=false)=>{status.textContent=message;status.dataset.error=String(error);};
 const setBusy=value=>{s.busy=value;button.textContent=value?'取消转写':'重新转写';tools.dataset.pending=String(value);};
 const cancelJob=id=>{if(id)void fetch('/api/jobs/'+encodeURIComponent(id)+'/cancel',{method:'POST'}).catch(()=>{});};
 function invalidate(){s.epoch++;clearTimeout(s.timer);cancelJob(s.job);s.job=null;setBusy(false);}
 function putText(text){s.writing=true;b.target.value=text;b.target.dispatchEvent(new Event('input',{bubbles:true}));s.writing=false;}
 b.target.addEventListener('input',()=>{if(!s.writing)s.revision++;});
 adopt.onclick=()=>{if(!proposal.value)return;putText(proposal.value);detail.open=false;setStatus('已采用转写，请试听后继续校稿。');};
 async function run(epoch,baseline,revision){
  const original=source();if(!original){setStatus('请先选择参考音频。',true);return;}
  setBusy(true);detail.hidden=true;setStatus('正在提交本地自动转写…');
  try{
   let file=original;if(typeof original==='string'){const u=new URL(original,location.origin);if(u.origin!==location.origin)throw Error('参考来源必须属于本地工作台');const r=await fetch(u.href);if(!r.ok)throw Error('参考原音不存在，请重新选择');file=new File([await r.blob()],'reference.wav');}
   if(epoch!==s.epoch)return;
   const form=new FormData();form.append('file',file,file.name);form.append('language','auto');if(b.start)form.append('start_seconds',b.start.value||'0');if(b.end)form.append('end_seconds',b.end.value||'');
   const queued=await json('/api/transcribe',{method:'POST',body:form});
   if(epoch!==s.epoch){cancelJob(queued.job_id);return;}s.job=queued.job_id;
   for(let poll=0;poll<1200;poll++){
    if(epoch!==s.epoch)return;const job=await json('/api/jobs/'+encodeURIComponent(queued.job_id));if(epoch!==s.epoch)return;
    if(job.status==='done'){
     const result=job.result||{},text=(result.text||result.transcript||'').trim();if(!text)throw Error('没有识别到文字，请试听原音后手工填写或重试');
     proposal.value=text;detail.hidden=false;
     const backend=result.backend==='r2t2'?'Confucius R2T2':result.backend==='qwen'?'Qwen3-ASR':'本地 ASR';
     if(s.revision===revision&&b.target.value===baseline&&source()===original){putText(text);setStatus('已自动填写 · '+backend+(result.truncated?' · 结果可能截断，请补齐校稿。':' · 请试听并校稿。'));}
     else{detail.open=true;setStatus('转写已完成，你编辑的文字已保留。可对照下方结果，或点击采用。');}
     return;
    }
    if(!['queued','running'].includes(job.status))throw Error(job.error||'转写已取消，可重新转写或手工填写');
    setStatus(job.status==='queued'?'自动转写已排队，等待当前任务完成…':'正在自动转写'+(Number(job.progress)>0?' · '+Math.round(Number(job.progress)*100)+'%':'')+'，台词框仍可手动编辑。');
    await new Promise(resolve=>setTimeout(resolve,850));
   }
   throw Error('转写等待时间较长，请在任务列表查看状态。');
  }catch(error){if(epoch===s.epoch)setStatus('转写未完成：'+error.message+'。可手动填写或重试。',true);}
  finally{if(epoch===s.epoch){s.job=null;setBusy(false);}}
 }
 function changed(event){
  invalidate();detail.hidden=true;
  if(event?.detail?.referenceTranscript){putText(event.detail.referenceTranscript);setStatus('已带入本地转写结果，可以继续手动校稿。');return;}
  if(!source()){setStatus('选择音频后自动转写。');return;}
  if(!toggle.checked){setStatus('自动转写已关闭，可手动填写或点击重新转写。');return;}
  const epoch=s.epoch,baseline=b.target.value,revision=s.revision;setBusy(true);setStatus('准备自动转写…');s.timer=setTimeout(()=>{s.timer=null;void run(epoch,baseline,revision);},250);
 }
 b.input.addEventListener('change',changed);for(const field of [b.start,b.end])field?.addEventListener('change',changed);
 toggle.onchange=()=>{if(toggle.checked)changed();else{invalidate();setStatus('自动转写已关闭，可手动填写或重新转写。');}};
 button.onclick=()=>{if(s.busy){invalidate();setStatus('已取消自动转写，当前文字保留。');return;}invalidate();void run(s.epoch,b.target.value,s.revision);};
 b.input.form?.addEventListener('submit',event=>{if(!s.busy)return;event.preventDefault();event.stopImmediatePropagation();setStatus('正在转写，请稍候再编码保存；也可取消转写后手工填写。',true);},true);
 b.input.form?.addEventListener('reset',()=>{invalidate();detail.hidden=true;setStatus('选择音频后自动转写。');});
}
})();
