import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let context=null;
const active=new Map();
document.addEventListener('pointerdown',()=>{
    const requested=app.graph?._nodes.some(n=>n.type==='T8DramaLive'&&n.widgets?.some(w=>w.name==='preview_audio'&&w.value));
    if(requested){context=context||new AudioContext();context.resume();}
});
app.registerExtension({name:'T8.Drama3.LiveAudio',setup(){
    api.addEventListener('drama3_t8_live_audio',async event=>{
        const data=event.detail;
        if(!data.preview||!context)return;
        const state=active.get(data.node_id)||{time:context.currentTime,players:new Set(),pending:Promise.resolve(),cancelled:false};
        active.set(data.node_id,state);
        state.pending=state.pending.then(async()=>{
            if(state.cancelled)return;
            const query=new URLSearchParams({filename:data.filename,subfolder:data.subfolder,type:data.type});
            const response=await api.fetchApi(`/view?${query}`);
            if(!response.ok)throw Error(`音频分片读取失败：${response.status}`);
            const buffer=await context.decodeAudioData(await response.arrayBuffer());
            if(state.cancelled)return;
            const player=context.createBufferSource();player.buffer=buffer;player.connect(context.destination);
            state.time=Math.max(state.time,context.currentTime+.05);player.start(state.time);state.time+=buffer.duration;
            state.players.add(player);player.onended=()=>state.players.delete(player);
        }).catch(error=>console.error('Drama 3 live preview',error));
    });
    api.addEventListener('drama3_t8_live_stop',event=>{
        const state=active.get(event.detail.node_id);if(!state)return;
        state.cancelled=true;
        for(const player of state.players){try{player.stop()}catch{}}
        active.delete(event.detail.node_id);
    });
}});
