/* Preview a local upload immediately; no upload, ASR or model load is needed. */
(() => {
"use strict";
const widgets=new Map();
const style=document.createElement('style');
style.textContent=`.upload-audio-preview{margin:10px 0 16px;padding:13px 14px;border:1px solid #f1dbe7;border-radius:12px;background:#fff9fc;font-family:Segoe UI,Microsoft YaHei UI,sans-serif}.upload-audio-preview[hidden]{display:none!important}.upload-audio-preview .upload-preview-meta{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:9px}.upload-audio-preview strong{overflow:hidden;text-overflow:ellipsis;font-size:12px;font-weight:600;color:#9e5a7d}.upload-audio-preview span{font-size:11px;color:#9a8497;white-space:nowrap}.upload-audio-preview audio{display:block;width:100%;height:36px}.upload-audio-preview p{font-size:11px;color:#b56c86;margin:8px 0 0}`;
document.head.append(style);
function attach(input){
 if(widgets.has(input))return widgets.get(input);
 const box=document.createElement('section');box.className='upload-audio-preview';box.hidden=true;box.setAttribute('aria-label','已选音频试听');
 const meta=document.createElement('div');meta.className='upload-preview-meta';
 const name=document.createElement('strong'),duration=document.createElement('span'),audio=document.createElement('audio'),error=document.createElement('p');
 audio.controls=true;audio.preload='metadata';error.hidden=true;meta.append(name,duration);box.append(meta,audio,error);
 // Do not put an interactive player inside the file input's label.
 (input.closest('label')||input).after(box);
 const widget={box,audio,name,duration,error,objectURL:null,file:null};widgets.set(input,widget);
 audio.addEventListener('loadedmetadata',()=>{duration.textContent=Number.isFinite(audio.duration)?audio.duration.toFixed(2)+' 秒':'时长待读取';});
 audio.addEventListener('error',()=>{if(!audio.getAttribute('src'))return;error.textContent='当前浏览器无法试听这个文件，请尝试 WAV 或 MP3。';error.hidden=false;});
 audio.addEventListener('play',()=>{for(const other of document.querySelectorAll('audio'))if(other!==audio)other.pause();});
 input.addEventListener('change',()=>show(input,input.files?.[0]||null));
 input.form?.addEventListener('reset',()=>queueMicrotask(()=>show(input,input.files?.[0]||null)));
 if(input.files?.[0])show(input,input.files[0]);return widget;
}
function show(input,file,options={}){
 if(typeof input==='string')input=document.getElementById(input);
 if(!input)return;const w=attach(input);w.file=file;w.audio.pause();w.audio.removeAttribute('src');w.audio.load();
 if(w.objectURL){URL.revokeObjectURL(w.objectURL);w.objectURL=null;}
 w.error.hidden=true;w.box.hidden=!file;
 if(!file)return;w.name.textContent=options.name||file.name||'参考音频';w.duration.textContent='读取时长…';
 if(typeof file==='string'){const url=new URL(file,location.origin);if(url.origin!==location.origin||!['http:','https:'].includes(url.protocol)){w.box.hidden=true;return;}w.audio.src=url.href;}
 else{w.objectURL=URL.createObjectURL(file);w.audio.src=w.objectURL;}
 w.audio.load();
}
for(const input of document.querySelectorAll('input[type="file"]')){
 if(input.id==='cd-record-file')continue; // Recording already has a full waveform/player.
 if(/audio|\.wav|\.mp3|\.flac|\.webm/.test(input.accept))attach(input);
}
window.previewReferenceUpload=show;
window.referencePreviewFile=input=>widgets.get(typeof input==='string'?document.getElementById(input):input)?.file||null;
window.addEventListener('pagehide',()=>{for(const w of widgets.values()){w.audio.pause();if(w.objectURL)URL.revokeObjectURL(w.objectURL);}});
})();
