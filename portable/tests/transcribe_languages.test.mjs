import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source=fs.readFileSync(new URL('../web/transcribe.js',import.meta.url),'utf8');
function languages(file,name){
 const text=fs.readFileSync(new URL(file,import.meta.url),'utf8');
 const values=text.match(new RegExp(`^${name} = [\\[(]([\\s\\S]*?)[\\])]`,'m'));
 return JSON.parse(`[${values[1]}]`);
}
const supported={qwen:languages('../drama3_t8/qwen3_asr/asr.py','LANGUAGES'),r2t2:languages('../drama3_t8/r2t2/native.py','SUPPORTED_LANGUAGES')};
async function harness(backend='qwen'){
 const nodes=new Map(),notices=[];
 class Element{
  constructor(){this.options=[];this.selected='';this.hidden=false;this.dataset={};this.elements={};}
  set innerHTML(value){this.html=value;const match=value.match(/<select name="language">(.*?)<\/select>/s);if(match){const select=get('language');select.replaceChildren(...[...match[1].matchAll(/<option value="([^"]+)"[^>]*>([^<]*)/g)].map(m=>new Option(m[2],m[1])));}}
  set value(value){this.selected=String(value);}
  get value(){return this.options.length?(this.options.some(o=>o.value===this.selected)?this.selected:''):this.selected;}
  replaceChildren(...options){this.options=options;this.selected=options[0]?.value||'';}
  add(option){this.options.push(option);if(this.options.length===1)this.selected=option.value;}
  append(){} insertBefore(){} before(){} removeAttribute(){}
 }
 class Option{constructor(text,value){this.text=text;this.value=String(value);}}
 const get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
 get('asr-form').elements={language:get('language'),file:{files:[]}};
 let status={settings:{asr_backend:backend},asr_languages:{backend,languages:['auto',...supported[backend]]}};
 const context={$:get,Option,document:{createElement:()=>new Element(),querySelector:()=>new Element()},window:{},
  showPage(){},setTimeout(){},notice:message=>notices.push(message),
  api:async path=>path==='/api/status'?structuredClone(status):[],observedJob:async()=>null};
 vm.createContext(context);vm.runInContext(source,context);await Promise.resolve();await Promise.resolve();
 return {get,context,notices,setStatus(next){status=next;},async poll(){await context.pollASR();},
  async switchBackend(next){status={settings:{asr_backend:next},asr_languages:{backend:next,languages:['auto',...supported[next]]}};await context.pollASR();}};
}
test('saved Qwen backend exposes auto and all actual 30 languages',async()=>{
 const state=await harness('qwen');
 assert.equal(supported.qwen.length,30);
 assert.deepEqual(state.get('language').options.map(o=>o.value),['auto',...supported.qwen]);
 assert.equal(state.get('language').value,'auto');
 assert.match(state.get('asr-engine').textContent,/Qwen3-ASR/);
});
test('saved R2T2 backend exposes exactly its own supported languages',async()=>{
 const state=await harness('r2t2');
 assert.deepEqual(state.get('language').options.map(o=>o.value),['auto',...supported.r2t2]);
 assert.ok(!state.get('language').options.some(o=>o.value==='Arabic'));
 assert.match(state.get('asr-engine').textContent,/R2T2/);
});
test('polls and backend changes preserve a still valid manual choice',async()=>{
 const state=await harness('qwen');state.get('language').value='English';
 await state.poll();assert.equal(state.get('language').value,'English');
 await state.switchBackend('r2t2');assert.equal(state.get('language').value,'English');
 await state.switchBackend('qwen');assert.equal(state.get('language').value,'English');
 assert.equal(state.get('language').options.length,31);
});
test('a language unsupported by the newly saved backend resets to auto',async()=>{
 const state=await harness('qwen');state.get('language').value='Arabic';
 await state.switchBackend('r2t2');assert.equal(state.get('language').value,'auto');
 await state.switchBackend('qwen');assert.equal(state.get('language').value,'auto');
});
test('unsubmitted environment dropdown never changes the active ASR list',async()=>{
 const state=await harness('r2t2');state.get('runtime-asr').value='qwen';
 await state.poll();assert.deepEqual(state.get('language').options.map(o=>o.value),['auto',...supported.r2t2]);
});
test('missing or mismatched capability metadata only offers universal auto',async()=>{
 const state=await harness('qwen');
 state.setStatus({settings:{asr_backend:'r2t2'},asr_languages:{backend:'qwen',languages:['auto','Arabic']}});
 await state.poll();assert.deepEqual(state.get('language').options.map(o=>o.value),['auto']);
 state.setStatus({settings:{asr_backend:'qwen'}});
 await state.poll();assert.deepEqual(state.get('language').options.map(o=>o.value),['auto']);
});
