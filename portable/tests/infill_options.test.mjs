import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('../web/infill-options.js',import.meta.url),'utf8');
function fixture(){
 const nodes=new Map();
 const get=id=>{if(!nodes.has(id))nodes.set(id,{value:'',src:'',options:[],add(option){this.options.push(option);}});return nodes.get(id);};
 get('infill-target-speaker').value='auto';get('take-index').value='-1';
 const context={$:get,window:{clearEditorPreview(){get('audio').src='';}},Option:class{constructor(label,value){this.label=label;this.value=value;}},
  showResult(job){get('audio').src=job.result.audio;get('session').value=job.result.session_id;}};
 vm.createContext(context);vm.runInContext(source,context);
 return {get,show:context.showResult,read:context.window.readInfillOptions,clear:context.window.clearEditorPreview};
}
function job(subtype='PCM_16'){return{result:{audio:'http://local/output.wav',session_id:'project-a',take_index:2,take_count:3,
 playback_view:{subtype,samples:40000,pcm_sha256:'a'.repeat(64),source:'b'.repeat(64)}}};}
test('binds actual FLOAT or PCM16 asset automatically without a format control',()=>{
 for(const format of ['FLOAT','PCM_16']){
  const state=fixture();state.show(job(format));
  assert.equal(state.read().expected_view.subtype,format);assert.equal(state.read().target_speaker,null);
  state.get('infill-target-speaker').value='15';assert.equal(state.read().target_speaker,15);
  assert.equal(state.get('infill-target-speaker').options.length,16);
 }
});
test('changed project, version, audio asset or dismissed selection cannot reuse old offsets',()=>{
 for(const changed of ['project','take','audio','clear']){
  const state=fixture();state.show(job());
  if(changed==='project')state.get('session').value='project-b';
  if(changed==='take')state.get('take-index').value='1';
  if(changed==='audio')state.get('audio').src='http://local/other.wav';
  if(changed==='clear')state.clear();
  assert.throws(state.read,/试听所选版本/);
 }
});
test('legacy jobs without asset identity need a fresh preview and invalid roles are rejected',()=>{
 const state=fixture(),legacy=job();delete legacy.result.playback_view;state.show(legacy);
 assert.throws(state.read,/试听所选版本/);
 state.show(job('FLOAT'));state.get('infill-target-speaker').value='16';assert.throws(state.read,/有效/);
});
test('actual form reads the bound selection before sending the infill request',()=>{
 const app=fs.readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
 assert.match(app,/if\(page==="infill"\)\{try\{Object\.assign\(payload,window\.readInfillOptions\(\)\)/);
 const html=fs.readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
 assert.match(html,/id="infill-target-speaker"/);assert.match(html,/src="\/infill-options.js"/);
});
