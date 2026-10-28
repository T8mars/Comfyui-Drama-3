import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../web/scene-options.js',import.meta.url),'utf8');
function harness(saved){
 const values=new Map(),events=new Map(),notices=[];
 const split={checked:false,addEventListener:(name,fn)=>events.set('split',fn)};
 const budget={value:'300',addEventListener:(name,fn)=>events.set('budget',fn)};
 const context={$:id=>id==='scene-split-long-text'?split:budget,window:{},notice:value=>notices.push(value),
  localStorage:{getItem:()=>saved,setItem:(key,value)=>values.set(key,value)}};
 vm.createContext(context);vm.runInContext(source,context);
 return{split,budget,events,values,notices,read:context.window.readSceneChunkOptions,setPage:context.window.setSceneChunkPage};
}
test('legacy default stays disabled and persisted scene policy restores separately',()=>{
 const initial=harness(null);assert.equal(initial.split.checked,false);assert.equal(initial.budget.disabled,true);
 const state=harness(JSON.stringify({split_long_text:true,chunk_bytes:101}));
 state.setPage(true);
 assert.equal(state.split.checked,true);assert.equal(state.budget.disabled,false);
 assert.equal(state.read().chunk_bytes,101);
 state.split.checked=false;state.events.get('split')();
 assert.equal(state.budget.disabled,true);
 assert.deepEqual(JSON.parse([...state.values.values()][0]),{split_long_text:false,chunk_bytes:101});
 state.split.checked=true;state.events.get('split')();state.setPage(false);
 assert.equal(state.budget.disabled,true); // Unused hidden fields cannot block TTS.
});
test('bad persisted state cannot silently set invalid booleans or fractional/nonfinite budgets',()=>{
 for(const value of ['{',JSON.stringify({split_long_text:'yes',chunk_bytes:100}),JSON.stringify({split_long_text:true,chunk_bytes:32.5})]){
  const state=harness(value);assert.equal(state.split.checked,false);assert.equal(state.read().chunk_bytes,300);
 }
 const state=harness(null);
 for(const value of ['NaN','Infinity','32.5','31','3001']){state.budget.value=value;assert.throws(state.read,/整数/);}
});
test('actual form wiring sends these scene options only from the scene page',()=>{
 const app=fs.readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
 assert.match(app,/if\(page==="scene"\)\{try\{Object\.assign\(payload,window\.readSceneChunkOptions\(\)\)/);
 assert.match(app,/catch\(error\)\{notice\(error.message\);return\}/);
 const html=fs.readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
 assert.match(html,/id="scene-split-long-text"/);assert.match(html,/src="\/scene-options.js"/);
});
