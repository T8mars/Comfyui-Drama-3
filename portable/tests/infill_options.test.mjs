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
 return {get,show:context.showResult,read:context.window.readInfillOptions,clear:context.window.clearEditorPreview,
  bind:context.window.bindInfillPlaybackSource,invalidate:context.window.invalidateInfillSource,ready:context.window.hasCurrentInfillSource};
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
  assert.throws(state.read,/来源音频/);
 }
});
test('legacy jobs without asset identity need a fresh preview and invalid roles are rejected',()=>{
 const state=fixture(),legacy=job();delete legacy.result.playback_view;state.show(legacy);
 assert.throws(state.read,/来源音频/);
 state.show(job('FLOAT'));state.get('infill-target-speaker').value='16';assert.throws(state.read,/有效/);
});
test('separate source preview authorizes editing while the right player holds another result',()=>{
 const state=fixture(),source=job('FLOAT');
 state.get('session').value=source.result.session_id;state.get('audio').src='http://local/old-result.wav';
 state.get('source-preview-audio').src=source.result.audio;
 state.bind(source.result,'source-preview-audio');
 assert.equal(state.ready(),true);assert.equal(state.read().expected_view,source.result.playback_view);
 state.get('audio').src='http://local/new-result.wav';
 assert.equal(state.read().expected_view,source.result.playback_view);
 state.get('source-preview-audio').src='http://local/other-source.wav';assert.equal(state.ready(),false);assert.throws(state.read,/来源音频/);
});

test('separate preview still rejects changed source session, take and explicit invalidation',()=>{
 for(const change of ['session','take','invalidate']){
  const state=fixture(),source=job();state.get('session').value=source.result.session_id;
  state.get('source-preview-audio').src=source.result.audio;state.bind(source.result,'source-preview-audio');
  if(change==='session')state.get('session').value='another-project';
  if(change==='take')state.get('take-index').value='1';
  if(change==='invalidate')state.invalidate();
  assert.equal(state.ready(),false);assert.throws(state.read,/来源音频/);
 }
});

test('actual form awaits source preparation and sends its exact playback identity',async()=>{
 const app=fs.readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
 const start=app.indexOf('$("generate-form").onsubmit=async e=>{'),end=app.indexOf('$("voice-form").onsubmit=',start);
 assert.ok(start>=0&&end>start);
 const nodes=new Map(),trace=[],requests=[],expected=job('FLOAT').result.playback_view;
 const get=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:false,elements:[],disabled:false});return nodes.get(id);};
 for(const [id,value] of Object.entries({session:'project-a','take-index':'-1',text:'完整稿件。','start-seconds':'4.95','end-seconds':'5.05',seed:'42','chunk-bytes':'300',temperature:'.7','top-p':'.7','top-k':'30','max-frames':'600'}))get(id).value=value;
 let prepared=false;
 const context={page:'infill',submissionPending:false,audioSelectionGeneration:0,$:get,notice(){},submitFeedback(){},refreshJobs:async()=>{},document:{querySelectorAll:()=>[]},
  api:async(path,options)=>{requests.push(JSON.parse(options.body));trace.push('submit');return {job_id:'new-job'};},
  window:{syncGenerationMode(){},prepareInfillSource:async()=>{await Promise.resolve();prepared=true;trace.push('prepared');},snapCurrentEditRange(){trace.push('snapped');},getReferenceSpeakers:()=>({}),
   readInfillOptions:()=>{assert.ok(prepared);trace.push('bound');return {target_speaker:null,expected_view:expected};},watchAudioJob(){}}};
 vm.createContext(context);vm.runInContext(app.slice(start,end),context);
 const event={preventDefault(){},currentTarget:get('generate-form')};
 const submitted=get('generate-form').onsubmit(event);
 event.currentTarget=null; // Browsers clear this field after the handler yields.
 await submitted;
 assert.deepEqual(trace,['prepared','snapped','bound','submit']);assert.equal(requests.length,1);
 assert.equal(requests[0].session_id,'project-a');assert.equal(requests[0].take_index,-1);assert.deepEqual(requests[0].expected_view,expected);
 assert.equal(requests[0].start_seconds,4.95);assert.equal(requests[0].end_seconds,5.05);assert.equal(get('generate').disabled,false);
});
