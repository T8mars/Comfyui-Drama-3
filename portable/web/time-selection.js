/* One rounding rule for word alignment, waveform drags and typed edit bounds. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.DramaTimeSelection=factory();})(typeof window==='object'?window:globalThis,function(){
 'use strict';
 function nearestValue(value,duration=Infinity,step=.05){value=Number(value);if(!Number.isFinite(value)||value<0||!(duration>0)||!(step>0))throw Error('请选择有效的音频时间。');value=Math.min(value,duration);let snapped=Math.round(value/step+1e-10)*step;if(Number.isFinite(duration)&&(snapped>duration||Math.abs(duration-value)<Math.abs(snapped-value)))snapped=Math.floor(duration*1e6)/1e6;return Number(snapped.toFixed(6));}
 function nearestRange(start,end,duration=Infinity,step=.05){
  start=Number(start);end=Number(end);duration=Number(duration);step=Number(step);
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<start||!Number.isFinite(step)||step<=0||!(duration>0))throw Error('请选择有效的音频起止时间。');
  let a=nearestValue(start,duration,step),b=nearestValue(end,duration,step);
  if(b<=a){if(a<duration)b=Math.min(duration,a+step);else a=Math.max(0,Math.floor((duration-1e-9)/step)*step);}
  return {start:Number(a.toFixed(6)),end:Number(b.toFixed(6)),step};
 }
 return {nearestRange,nearestValue};
});
