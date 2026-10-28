// Scene-only request options. Runtime settings and other generation modes do
// not silently overwrite a saved scene's revision policy.
(() => {
 const key='drama-scene-chunk-options-v1',split=$('scene-split-long-text'),budget=$('scene-chunk-bytes');
 let active=false;
 function read(){
  const value=Number(budget.value);
  if(!Number.isInteger(value)||value<32||value>3000)throw Error('场景每次新剧本预算须为32–3000的整数');
  return {split_long_text:split.checked,chunk_bytes:value};
 }
 function update(){budget.disabled=!active||!split.checked;}
 try{
  const saved=JSON.parse(localStorage.getItem(key)||'null');
  if(saved&&typeof saved.split_long_text==='boolean'&&Number.isInteger(saved.chunk_bytes)&&saved.chunk_bytes>=32&&saved.chunk_bytes<=3000){
   split.checked=saved.split_long_text;budget.value=saved.chunk_bytes;
  }
 }catch{}
 function persist(){update();try{localStorage.setItem(key,JSON.stringify(read()));}catch(error){notice(error.message);}}
 split.addEventListener('change',persist);budget.addEventListener('change',persist);update();
 window.readSceneChunkOptions=read;
 window.setSceneChunkPage=value=>{active=value===true;update();};
})();
