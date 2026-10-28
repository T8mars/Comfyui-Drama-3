// Bind the time selection to the PCM16 asset the user actually previewed.
(() => {
 let selected=null;
 const role=$('infill-target-speaker');
 for(let index=0;index<16;index++)role.add(new Option(`角色 ${index}`,String(index)));
 const previous=showResult;
 showResult=function(job){
  previous(job);
  const result=job.result;
  if(!result?.audio)return;
  selected=result.playback_view?{session:result.session_id,index:result.take_index,count:result.take_count,
   audio:$('audio').src,view:result.playback_view}:null;
 };
 const clear=window.clearEditorPreview;
 window.clearEditorPreview=function(){selected=null;return clear?.apply(this,arguments);};
 window.readInfillOptions=()=>{
  const index=Number($('take-index').value),current=selected;
  if(!current||$('session').value!==current.session||$('audio').src!==current.audio||
    !(index===current.index||index===-1&&current.index===current.count-1)){
   throw Error('请先点击“试听所选版本”，确认当前工程与音频后再选区补写。');
  }
  const value=role.value;
  if(value!=='auto'&&!/^(?:[0-9]|1[0-5])$/.test(value))throw Error('请选择有效的目标角色');
  return {target_speaker:value==='auto'?null:Number(value),expected_view:current.view};
 };
})();
