let streamJob=null,streamGeneration=0,streamSeen=new Set(),streamContext=null,streamTime=0,streamPlayers=new Set();
window.cancelLiveJob=function(jobId){
  if(jobId!==streamJob)return;
  streamGeneration++;streamJob=null;
  for(const player of streamPlayers){try{player.stop()}catch{}}
  streamPlayers.clear();
};
window.stopLivePreview=function(){window.cancelLiveJob(streamJob);};
document.getElementById('generate-form').addEventListener('submit',()=>{
  window.cancelLiveJob(streamJob);
  if(!document.getElementById('audio-stream').checked)return;
  streamContext=streamContext||new AudioContext();streamContext.resume();
});
window.startLiveJob=function(jobId){
  window.cancelLiveJob(streamJob);
  streamGeneration++;streamJob=jobId;streamSeen.clear();
  if(streamContext)streamTime=streamContext.currentTime;
};
async function streamJSON(url){const response=await fetch(url);if(!response.ok)throw Error('音频预览连接失败');return response.json();}
async function livePoll(){
  const jobId=streamJob,generation=streamGeneration,current=()=>jobId===streamJob&&generation===streamGeneration;
  if(jobId&&streamContext){
    try{
      const chunks=await streamJSON(`/api/jobs/${jobId}/chunks`);
      for(const url of chunks){
        if(!current())break;
        if(streamSeen.has(url))continue;
        const response=await fetch(url);if(!response.ok)throw Error('音频分包暂不可用，将重试');
        const buffer=await streamContext.decodeAudioData(await response.arrayBuffer());
        if(!current())break;
        const player=streamContext.createBufferSource();player.buffer=buffer;player.connect(streamContext.destination);
        streamPlayers.add(player);player.onended=()=>streamPlayers.delete(player);
        streamTime=Math.max(streamTime,streamContext.currentTime+.05);player.start(streamTime);streamTime+=buffer.duration;
        streamSeen.add(url);
      }
      if(current()){
        const jobs=await streamJSON('/api/jobs'),job=jobs.find(j=>j.id===jobId)||await streamJSON(`/api/jobs/${encodeURIComponent(jobId)}`);
        if(current()&&job&&['failed','cancelled','interrupted'].includes(job.status))window.cancelLiveJob(jobId);
        if(current()&&job?.status==='done'){
          // Include a final chunk flushed between the first list and task completion.
          const finalChunks=await streamJSON(`/api/jobs/${jobId}/chunks`);
          if(current()&&finalChunks.every(url=>streamSeen.has(url)))streamJob=null;
        }
      }
    }catch(error){if(current())document.getElementById('task-status').textContent=`实时预览：${error.message}`}
  }
  setTimeout(livePoll,1200);
}livePoll();
