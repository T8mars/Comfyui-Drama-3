import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import fs from 'node:fs';

const source=fs.readFileSync(new URL('../web/workspace-tools.js',import.meta.url),'utf8');
const keys=['model','codec','text_head','line_head','asr_model','aligner','r2t2_model','r2t2_projector','r2t2_vad'];
async function workspace(){
 const nodes=new Map(),created=[],requests=[];
 class Element{
  constructor(){this.value='';this.options=[];this.classList={toggle(){}};this.checked=false;}
  set innerHTML(html){this.html=html;for(const match of html.matchAll(/<(?:input|select)[^>]*id="([^"]+)"[^>]*>/g)){const el=get(match[1]);el.required=/\srequired(?:\s|>)/.test(match[0]);}}
  get innerHTML(){return this.html;}
  append(){} prepend(){} add(v){this.options.push(v);} addEventListener(){} replaceChildren(){} setAttribute(){}
 }
 const get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
 const settings={...Object.fromEntries(keys.map(k=>[k,k+'.safetensors'])),expert_placement:'vram',asr_backend:'r2t2',gdn_backend:'torch',expert_cache_gb:8,moe_backend:'auto',cuda_graph:false};
 const context={document:{createElement(){const el=new Element();created.push(el);return el;},querySelectorAll:()=>[]},
  $:get,window:{},localStorage:{getItem:()=>null},setTimeout:()=>{},Option:class{},showResult(){},notice(){},
  api:async(path,options)=>{if(options){requests.push(JSON.parse(options.body));return {job_id:'job'};}return {settings,model_paths:{},acceleration:{}};},
  observedJob:async()=>null,audioSelectionGeneration:0};
 vm.createContext(context);vm.runInContext(source,context);await Promise.resolve();await Promise.resolve();
 return {get,requests,form:created.find(el=>el.className==='runtime-form'),context};
}
test('runtime form requires core files and allows empty unused auxiliary inputs',async()=>{
 const state=await workspace();
 for(const key of keys)assert.equal(state.get('runtime-'+key).required,['model','codec'].includes(key),key);
 for(const key of keys)if(!['model','codec','line_head'].includes(key))state.get('runtime-'+key).value='';
 await state.form.onsubmit({preventDefault(){}});
 const saved=state.requests[0].settings;
 assert.equal(saved.aligner,'');assert.equal(saved.r2t2_vad,'');assert.equal(saved.asr_model,'');
 assert.ok(saved.model);assert.ok(saved.codec);
});
test('actual settings load and submit preserve explicit placement and graph selection',async()=>{
 const state=await workspace();assert.equal(state.get('runtime-placement').value,'vram');
 assert.equal(state.get('runtime-cache').disabled,true);
 state.get('runtime-placement').value='ram';
 state.get('runtime-placement').onchange();
 assert.equal(state.get('runtime-cache').disabled,false);
 await state.form.onsubmit({preventDefault(){}});
 assert.equal(state.requests[0].settings.expert_placement,'ram');
 assert.equal(state.requests[0].settings.cuda_graph,false);
 assert.match(state.context.describeAcceleration({expert_placement:'vram',moe_backend:'not_run'}),/单卡显存.*尚未运行/);
});
