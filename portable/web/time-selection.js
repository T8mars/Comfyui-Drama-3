/* One rounding rule for word alignment, waveform drags and typed edit bounds. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.DramaTimeSelection=factory();})(typeof window==='object'?window:globalThis,function(){
 'use strict';
 function nearestValue(value,duration=Infinity,step=.05){value=Number(value);if(!Number.isFinite(value)||value<0)throw Error('请选择有效的音频时间。');const max=Number.isFinite(duration)?Math.floor((duration+1e-9)/step):Infinity;return Number((Math.min(max,Math.round(value/step+1e-10))*step).toFixed(6));}
 function nearestRange(start,end,duration=Infinity,step=.05){
  start=Number(start);end=Number(end);duration=Number(duration);step=Number(step);
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<start||!Number.isFinite(step)||step<=0||!(duration>0))throw Error('请选择有效的音频起止时间。');
  const max=Number.isFinite(duration)?Math.floor((duration+1e-9)/step):Infinity;
  if(max<1)throw Error('原音频过短，无法选择有效区间。');
  let a=Math.max(0,Math.min(max,Math.round(start/step+1e-10))),b=Math.max(0,Math.min(max,Math.round(end/step+1e-10)));
  if(b<=a){if(a<max)b=a+1;else a=Math.max(0,b-1);}
  return {start:Number((a*step).toFixed(6)),end:Number((b*step).toFixed(6)),step};
 }
 return {nearestRange,nearestValue};
});
