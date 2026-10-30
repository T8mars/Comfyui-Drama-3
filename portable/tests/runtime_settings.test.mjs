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
test('release display distinguishes returned pinned RAM from residual process memory',async()=>{
 const state=await workspace(),GiB=1024**3;
 const text=state.context.describeUnloadMemory({status:'ok',before:{host_allocator:{status:'measured','allocated_bytes.current':17*GiB},process:{rss:20*GiB}},after:{host_allocator:{status:'measured','allocated_bytes.current':0},process:{rss:9*GiB},owned_objects:{model:{weakref_supported:true,alive:false},codec:{weakref_supported:true,alive:false}}}});
 assert.match(text,/固定内存 17.0 → 0.0 GiB/);assert.match(text,/进程内存 20.0 → 9.0 GiB/);assert.match(text,/2\/2 已清理/);
});
test('missing release observations never become invented zero memory or cleared objects',async()=>{
 const state=await workspace();assert.equal(state.context.describeUnloadMemory(null),'');
 const text=state.context.describeUnloadMemory({status:'ok',after:{host_allocator:{status:'unavailable'},owned_objects:{unknown:{weakref_supported:false,alive:null}}}});
 assert.match(text,/暂无统计/);assert.doesNotMatch(text,/0.0|已清理/);
});
test('refused and failed unload reports do not advertise successful cleanup',async()=>{
 const state=await workspace();assert.match(state.context.describeUnloadMemory({status:'refused_active'}),/模型保持加载/);
 assert.match(state.context.describeUnloadMemory({status:'error'}),/未完整结束/);
});
