// Same source-bound editor and planner as T8DramaEditor; no external service.
import { EditorView, EditorWindow, TakePreview, injectStyle } from './editor_core.js';

injectStyle();
const byId=(id)=>document.getElementById(id);
const request=async(url,body)=>{
  const r=await fetch(url,body===undefined?undefined:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await r.json();
  if(!r.ok)throw Error(data.detail||data.error||`HTTP ${r.status}`);
  return data;
};
const box=document.createElement('section');
box.className='studio-post-editor';
box.innerHTML='<h3>多轨后期剪辑</h3><p>分角色调整音量、剪去片段、插入静音或压缩停顿。应用时保存为新工程。</p><label><input type="checkbox" id="post-align-words">准备时启用词级字幕（使用设置中的对齐模型）</label><button type="button" class="secondary" id="post-editor-open">打开当前演出的多轨编辑器</button><p id="post-editor-message" role="status"></p><div id="post-editor-preview"></div>';
document.querySelector('.output-panel').append(box);
const style=document.createElement('style');
style.textContent='.studio-post-editor{border-top:1px solid #f2dce7;margin-top:20px;padding-top:16px}.studio-post-editor h3{font-size:15px}.studio-post-editor p{font-size:12px;color:#957284;line-height:1.6}.studio-post-editor label{font-size:12px;display:flex;gap:7px;align-items:center;margin:10px 0}.studio-post-editor input{width:auto}.studio-post-editor .d3e-node{min-height:220px;margin-top:12px}';
document.head.append(style);
let generation=0,view=null,preview=null,window_=null,current=null;
const message=(text)=>{byId('post-editor-message').textContent=text;};
const selection=()=>({id:byId('session').value,index:Number(byId('take-index')?.value??-1)});
const matches=(selected)=>{const now=selection();return now.id===selected.id&&now.index===selected.index;};
function dispose(){generation++;current=null;window_?.close();window_=null;preview?.destroy();preview=null;view?.destroy();view=null;}
for(const id of ['session','take-index'])byId(id)?.addEventListener('change',()=>{dispose();message('所选演出已改变，请重新打开。');});
async function openEditor(){
  const selected=selection();
  if(!selected.id){message('请先生成或导入含音频的工程。');return;}
  dispose();const token=generation;
  const stillSelected=()=>{
    if(token!==generation)return false;
    if(matches(selected))return true;
    dispose();message('所选演出已改变，请重新打开。');return false;
  };
  const button=byId('post-editor-open');button.disabled=true;message('正在准备原始音频与时间轴…');
  try{
    const submitted=await request('/api/jobs',{mode:'editor_prepare',session_id:selected.id,take_index:selected.index,
      align_words:byId('post-align-words').checked,language:byId('align-language')?.value||'Chinese'});
    let job;
    while(token===generation&&matches(selected)){
      job=await request(`/api/jobs/${submitted.job_id}`);
      if(!['queued','running'].includes(job.status))break;
      await new Promise((resolve)=>setTimeout(resolve,650));
    }
    if(!stillSelected())return;
    if(job.status!=='done')throw Error(job.error||'准备任务已取消');
    current={...job.result.editor,selected};
    const key=`drama3.t8.editor.${current.analysis.source}`;
    let draft=current.edits;
    try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved?.source===current.analysis.source)draft=saved;}catch{}
    const active=current;
    const activeView=new EditorView({readEdits:()=>draft,writeEdits:(edits)=>{draft=structuredClone(edits);try{localStorage.setItem(key,JSON.stringify(edits));}catch{}},
      plan:(analysis,edits)=>request('/api/editor/plan',{source_id:active.source_id,edits}),
      apply:async()=>{
        if(current!==active||!matches(selected))throw Error('所选工程或演出已改变，请重新打开。');
        message('正在保存剪辑后的新工程…');
        const result=await request('/api/editor/render',{source_id:active.source_id,session_id:active.session_id,take_index:active.take_index,edits:draft});
        if(current!==active||!matches(selected))return;
        window.showResult?.({id:active.source_id,result});
        const takeSelect=byId('take-index');
        if(takeSelect){
          if(![...takeSelect.options].some(o=>o.value===String(result.take_index)))takeSelect.add(new Option(`第 ${result.take_index+1} 遍 · 后期剪辑`,result.take_index));
          takeSelect.value=String(result.take_index);
        }
        window.refreshStatus?.();
        message('已保存新工程，原始演出和模型历史保留。音频、字幕和剪辑记录均可下载。');
        dispose();
      },applyText:{label:'✓ 保存剪辑为新工程',title:'保存音频、字幕、参数和工程；原始工程保留（Ctrl+Enter）',done:'正在保存新工程…'}});
    view=activeView;
    preview=new TakePreview(activeView,{emptyText:'正在载入原始音频…',onOpen:()=>{if(!window_)window_=new EditorWindow(activeView,()=>{window_=null;});}});
    byId('post-editor-preview').append(preview.root);
    const response=await fetch(active.audio);
    if(!response.ok)throw Error('原始音频预览失效，请重新打开。');
    if(!stillSelected())return;
    const bytes=await response.arrayBuffer();
    if(!stillSelected())return;
    const buffer=await activeView.player.context().decodeAudioData(bytes);
    if(!stillSelected())return;
    await activeView.load(active.analysis,buffer);
    if(!stillSelected())return;
    preview.setStatus(null);
    window_=new EditorWindow(activeView,()=>{window_=null;});
    message(active.analysis.aligned?'已启用词级字幕。':active.analysis.lines.some(line=>line.needs_alignment)?
      '补写后字幕需要完整稿件对齐；请启用词级字幕重新准备，当前未推测中间文字。':
      '字幕按句显示；启用词级字幕可按实际字词边界检查和剪切。');
  }catch(error){if(token===generation)message(error.message);}
  finally{if(token===generation||!current)button.disabled=false;}
}
byId('post-editor-open').onclick=openEditor;
