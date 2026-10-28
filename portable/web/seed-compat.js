import { app } from '../../scripts/app.js';

// v0.1.11 canvas layouts, including the frontend's former serialized control.
// Keep the layouts explicit: never treat an arbitrary string as a seed control.
const legacyLayouts={
  T8DramaGenerate:['mode','text','lookahead','chunk_bytes','seed','control_after_generate','temperature','top_p','top_k','max_frames','seed_text'],
  T8DramaScene:['script','scene','characters','seed','control_after_generate','max_frames','seed_text'],
  T8DramaInfill:['target_transcript','start_seconds','end_seconds','duration_frames','seed','control_after_generate','max_frames','seed_text'],
  T8DramaSRT:['srt','seed','control_after_generate','max_frames','seed_text'],
  T8DramaLive:['text','preview_audio','seed','control_after_generate','max_frames','seed_text'],
};
const controls=new Set(['fixed','increment','decrement','randomize']);
const property='drama3_t8_seed_control';
const pending=new WeakMap();

function readLegacyControl(nodeData,info,widgets){
  if(!Array.isArray(info?.widgets_values))return;
  const knownNames=new Set([
    ...Object.keys(nodeData.input?.required??{}),...Object.keys(nodeData.input?.optional??{}),
    ...Object.keys(nodeData.inputs??{}),
  ]);
  const seed=widgets.find(widget=>widget.name==='seed');
  const current=widgets.filter(widget=>knownNames.has(widget.name)||
    widget.name==='control_after_generate'&&seed?.linkedWidgets?.includes(widget)).map(widget=>widget.name);
  const legacy=legacyLayouts[nodeData.name];
  // The source update before scene chunk controls already appended this
  // selector. Its exact, known layout must also survive two later inputs.
  const priorScene=nodeData.name==='T8DramaScene'?[...legacy,'generation_mode']:null;
  const chunkScene=nodeData.name==='T8DramaScene'?[...legacy,'generation_mode','split_long_text','chunk_bytes']:null;
  for(const layout of [legacy,priorScene,chunkScene,current]){
    if(!layout||layout.length!==info.widgets_values.length)continue;
    if(layout.includes('generation_mode')&&!['together','per_line','per_line_context','per_speaker'].includes(info.widgets_values[layout.indexOf('generation_mode')]))continue;
    const index=layout.indexOf('control_after_generate');
    if(index<0||!controls.has(info.widgets_values[index]))continue;
    const value=info.widgets_values[layout.indexOf('seed')];
    const text=info.widgets_values[layout.indexOf('seed_text')];
    if(!Number.isSafeInteger(value)||value<0||typeof text!=='string')continue;
    const exact=text.trim();
    if(!/^\d{0,20}$/.test(exact)||exact&&BigInt(exact)>18446744073709551615n)continue;
    return {value:info.widgets_values[index],layout,current};
  }
}

app.registerExtension({
  name:'T8Drama3.SeedControlCompatibility',
  beforeRegisterNodeDef(nodeType,nodeData){
    if(!legacyLayouts[nodeData.name])return;
    // New frontend configure() removes nonserialized linked controls before
    // onConfigure runs. Capture the recognized legacy value before that step.
    const configureNode=nodeType.prototype.configure;
    if(typeof configureNode==='function')nodeType.prototype.configure=function(info){
      const saved=info?.properties?.[property];
      const legacy=readLegacyControl(nodeData,info,this.widgets??[]);
      const wanted=controls.has(saved)?saved:legacy?.value;
      if(legacy&&legacy.current.length>legacy.layout.length){
        // Official migration cannot recognize a shorter v0.1.11 array once new
        // inputs have been appended. Fill only newly introduced inputs from
        // the actual node's defaults, retaining every old value by name.
        const old=Object.fromEntries(legacy.layout.map((name,index)=>[name,info.widgets_values[index]]));
        const defaults=new Map(this.widgets.map(widget=>[widget.name,widget.value]));
        info.widgets_values=legacy.current.map(name=>Object.hasOwn(old,name)?old[name]:defaults.get(name));
      }
      const previous=pending.get(this);
      pending.set(this,{wanted});
      try{return configureNode.apply(this,arguments);}
      finally{if(previous)pending.set(this,previous);else pending.delete(this);}
    };
    const configure=nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure=function(info){
      const control=this.widgets?.find(widget=>widget.name==='control_after_generate');
      const before=control?.value;
      const result=configure?.apply(this,arguments);
      if(!control)return result;
      const saved=info?.properties?.[property];
      const wanted=pending.get(this)?.wanted??(controls.has(saved)?saved:readLegacyControl(nodeData,info,this.widgets)?.value);
      // Respect a prior onConfigure hook that deliberately chose another value.
      if(wanted!==undefined&&(control.value===before||control.value===wanted))control.value=wanted;
      return result;
    };
    const serialize=nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize=function(info){
      const result=serialize?.apply(this,arguments);
      const value=this.widgets?.find(widget=>widget.name==='control_after_generate')?.value;
      if(info&&controls.has(value)){
        info.properties??={};
        info.properties[property]=value;
      }
      return result;
    };
  },
});
