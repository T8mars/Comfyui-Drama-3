const samplingPreset=document.getElementById('sampling-preset');
const samplingInputs=['temperature','top-p','top-k'].map(id=>document.getElementById(id));
samplingPreset.addEventListener('change',()=>{
    const values={stable:[.7,.7,30],expressive:[1.,.9,0]}[samplingPreset.value];
    if(values)samplingInputs.forEach((input,i)=>input.value=values[i]);
});
samplingInputs.forEach(input=>input.addEventListener('input',()=>samplingPreset.value='custom'));
