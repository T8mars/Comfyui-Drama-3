// Bind edits to the selected source asset, independently of the result player.
(() => {
 let selected=null;
 const role=$('infill-target-speaker');
 for(let index=0;index<16;index++)role.add(new Option(`角色 ${index}`,String(index)));
 window.invalidateInfillSource=()=>{selected=null;};
 window.bindInfillPlaybackSource=(result,player='audio')=>{
  selected=result?.audio&&result.playback_view?{session:result.session_id,index:result.take_index,count:result.take_count,
   player,audio:$(player).src,view:result.playback_view}:null;
 };
 function currentSource(){
  const index=Number($('take-index').value),current=selected;
  return current&&current.audio&&$('session').value===current.session&&$(current.player).src===current.audio&&
   (index===current.index||index===-1&&current.index===current.count-1)?current:null;
 }
 window.hasCurrentInfillSource=()=>Boolean(currentSource());
 const previous=showResult;
 showResult=function(job){
  previous(job);
  const result=job.result;
  if(!result?.audio)return;
  window.bindInfillPlaybackSource(result,'audio');
 };
 const clear=window.clearEditorPreview;
 window.clearEditorPreview=function(){selected=null;return clear?.apply(this,arguments);};
 window.readInfillOptions=()=>{
  const current=currentSource();
  if(!current){
   throw Error('所选来源音频尚未准备好，或会话与版本已改变，请稍候再生成。');
  }
  const value=role.value;
  if(value!=='auto'&&!/^(?:[0-9]|1[0-5])$/.test(value))throw Error('请选择有效的目标角色');
  return {target_speaker:value==='auto'?null:Number(value),expected_view:current.view};
 };
 const loaded=window.getEditorSourcePreview?.();
 if(loaded)window.bindInfillPlaybackSource(loaded,'source-preview-audio');
})();
