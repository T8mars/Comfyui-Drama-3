import test from 'node:test';
import assert from 'node:assert/strict';
import {EditorView,loadTake} from '../web/editor_core.js';

globalThis.cancelAnimationFrame=()=>{};
const analysis=(source='a'.repeat(64))=>({source,duration:2,aligned:false,speakers:[0,1],lines:[
  {speaker:0,start:0,end:1,text:'Cat',tokens:[{text:'Cat',spoken:true}]},
  {speaker:1,start:1,end:2,text:'Dog',tokens:[{text:'Dog',spoken:true}]}]});
const buffer={sampleRate:100,numberOfChannels:1,length:200,getChannelData:()=>Float32Array.from({length:200},(_,i)=>i<100?.2:.4)};
const plan=async(a,e)=>({duration:2,pieces:[{src:[0,1],line:0},{src:[1,2],line:1}],pauses:[]});
const viewOf=(saved={},planner=plan)=>{
  let stored=saved,applied=0;
  const view=new EditorView({readEdits:()=>stored,writeEdits:e=>{stored=e},plan:planner,apply:()=>{applied++}});
  return{view,get stored(){return stored},get applied(){return applied}};
};
test('load defaults preserve raw audio settings and bind source; changed source drops draft',async()=>{
  const state=viewOf({source:'b'.repeat(64),cuts:[[0,.5]],normalize:true});
  await state.view.load(analysis(),buffer);
  assert.deepEqual(state.stored.cuts,[]);assert.equal(state.stored.trim_start,false);assert.equal(state.stored.normalize,false);
  assert.equal(state.stored.source,'a'.repeat(64));assert.equal(state.view.plan.duration,2);
});
test('same-source workflow draft survives loading; undo/redo preserve binding',async()=>{
  const state=viewOf({source:'a'.repeat(64),cuts:[[0,.25]]});
  await state.view.load(analysis(),buffer);
  state.view.change({inserts:[[1,.5]]});await Promise.resolve();
  assert.deepEqual(state.stored.cuts,[[0,.25]]);state.view.undo();await Promise.resolve();
  assert.deepEqual(state.stored.inserts,[]);state.view.redo();await Promise.resolve();
  assert.deepEqual(state.stored.inserts,[[1,.5]]);assert.equal(state.stored.source,'a'.repeat(64));
});
test('stale network plan cannot overwrite latest timeline',async()=>{
  const state=viewOf();await state.view.load(analysis(),buffer);
  const pending=[];state.view.requestPlan=()=>new Promise(resolve=>pending.push(resolve));
  const old=state.view.replan();const latest=state.view.replan();
  pending[1]({duration:3,pieces:[{gap:3,line:0}],pauses:[]});await latest;
  pending[0]({duration:99,pieces:[{gap:99,line:0}],pauses:[]});await old;
  assert.equal(state.view.plan.duration,3);
});
test('failure of a superseded request cannot replace a newer success with an error',async()=>{
  const state=viewOf();await state.view.load(analysis(),buffer);
  const pending=[];state.view.requestPlan=()=>new Promise((resolve,reject)=>pending.push({resolve,reject}));
  state.view.change({lead:.1});state.view.change({lead:.2});
  pending[1].resolve({duration:2,pieces:[{src:[0,2],line:0}],pauses:[]});await Promise.resolve();
  pending[0].reject(Error('Old source'));await Promise.resolve();await Promise.resolve();
  assert.equal(state.view.error,null);assert.equal(state.view.edits.lead,.2);
});
test('normalization includes per-speaker gain; original bypasses gain and normalization',async()=>{
  const state=viewOf();await state.view.load(analysis(),buffer);
  state.view.change({normalize:true,speakers:{'0':{gain_db:6},'1':{mute:true}}});await Promise.resolve();
  assert.ok(Math.abs(state.view.normalization*Math.pow(10,6/20)*.2-Math.pow(10,-1/20))<1e-6);
  let gain;state.view.player.play=(_layout,_buffer,_from,_stop,fn)=>{gain=fn};state.view.tick=()=>{};
  state.view.setOriginal(true);state.view.play();assert.equal(gain(0),1);assert.equal(gain(1),1);
});
test('planning failure blocks apply and exposes actionable error',async()=>{
  const state=viewOf();await state.view.load(analysis(),buffer);
  state.view.requestPlan=async()=>{throw Error('Source changed')};await state.view.apply();
  assert.equal(state.applied,0);assert.equal(state.view.error,'Source changed');assert.equal(state.view.applying,false);
});
test('newest load wins when audio fetch responses arrive out of order',async()=>{
  const state=viewOf();const pending=[];
  globalThis.fetch=url=>new Promise(resolve=>pending.push({url,resolve}));
  state.view.player.context=()=>({decodeAudioData:async()=>buffer});
  const first=loadTake(state.view,'/old-analysis','/old-audio');const second=loadTake(state.view,'/new-analysis','/new-audio');
  const resolve=(name,data)=>pending.find(p=>p.url===name).resolve({ok:true,json:async()=>data,arrayBuffer:async()=>new ArrayBuffer(0)});
  resolve('/new-analysis',analysis('b'.repeat(64)));resolve('/new-audio');assert.equal(await second,true);
  resolve('/old-analysis',analysis());resolve('/old-audio');assert.equal(await first,false);
  assert.equal(state.view.take.analysis.source,'b'.repeat(64));
});
