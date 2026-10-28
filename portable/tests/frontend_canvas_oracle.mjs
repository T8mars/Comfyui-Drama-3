// Execute the installed ComfyUI frontend's own ordering, INT widget and legacy
// restoration functions. No model, server or browser is started by this oracle.
import fs from 'node:fs';
import vm from 'node:vm';

const request=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const assets=fs.readdirSync(request.assets).filter(name=>name.endsWith('.js'));
const priority=name=>/^(settingStore|nodeDefStore|executionErrorStore)-/.test(name)?1:0;
const orderedAssets=assets.sort((left,right)=>priority(right)-priority(left));
const loaded=new Map();
let source;
for(const name of orderedAssets){
  const path=request.assets+'/'+name,text=fs.readFileSync(path,'utf8');
  if(priority(name))loaded.set(path,{name,path,text});
  if(text.includes('function getOrderedInputSpecs(')&&text.includes('function migrateWidgetsValues(')){
    source={name,path,text};break;
  }
}
if(!source)throw Error('Installed frontend ordering/restoration implementation was not found');

function bodyAt(marker,optional=false){
  let found,start;
  const match=file=>typeof marker==='string'?file.text.indexOf(marker):file.text.search(marker);
  for(const file of [source,...loaded.values()])if((start=match(file))>=0){found=file;break;}
  if(!found)for(const name of orderedAssets){
    const path=request.assets+'/'+name;
    if(loaded.has(path))continue;
    const file={name,path,text:fs.readFileSync(path,'utf8')};
    if((start=match(file))>=0){found=file;loaded.set(path,file);break;}
  }
  if(!found){if(optional)return;throw Error('Installed frontend function missing: '+marker);}
  const text=found.text;
  let parameters=text.indexOf('(',start),nested=0,parameterEnd;
  for(let index=parameters;index<text.length;index++){
    if(text[index]==='(')nested++;
    if(text[index]===')'&&!--nested){parameterEnd=index;break;}
  }
  const open=text.indexOf('{',parameterEnd+1);
  let depth=0,quote=null,escape=false;
  for(let index=open;index<text.length;index++){
    const character=text[index];
    if(quote){if(escape){escape=false;continue;}if(character==='\\'){escape=true;continue;}if(character===quote)quote=null;continue;}
    if(character==='"'||character==="'"||character==='`'){quote=character;continue;}
    if(character==='{')depth++;
    if(character==='}'&&!--depth)return text.slice(start,index+1);
  }
  throw Error('Unterminated installed frontend function '+marker);
}
const context=vm.createContext({
  he:(items,predicate)=>items.filter(predicate),fe:(items,fn)=>items.map(fn),
  $:()=>({get:()=>false}),isIntInputSpec:spec=>spec.type==='INT',
  onValueChange(){},transformInputSpecV2ToV1:spec=>[spec.type,spec],
  jg:Symbol(),Sx:Symbol(),isComboWidget$2:()=>false,updateControlWidgetLabel(){},
  controlValueRunBefore$1:()=>false,nextValueForLinkedTarget:()=>undefined,
});
const functionCode=name=>bodyAt(new RegExp('function\\s+'+name+'(?:\\$\\d+)?\\('));
for(const name of ['getOrderedInputSpecs','migrateWidgetsValues','computeLegacyWidgetShadow','serialiseWidgetValues','addValueControlWidget','addValueControlWidgets']){
  vm.runInContext(functionCode(name),context);
}
if(functionCode('serialiseWidgetValues').includes('cloneWidgetValueTwice('))vm.runInContext(bodyAt('function cloneWidgetValueTwice('),context);
if(functionCode('addValueControlWidgets').includes('controlAfterGenerateNameOverride(')){
  context.isValueControlMode=value=>['fixed','increment','decrement','randomize','increment-wrap'].includes(value);
  vm.runInContext(bodyAt('function controlAfterGenerateNameOverride('),context);
}
const migrationSource=functionCode('migrateWidgetsValues');
const helpers=migrationSource.match(/new Set\(([\w$]+)\(([\w$]+)\(t,e=>e\.serialize===!1\),e=>e\.name\)\)/);
if(helpers){context[helpers[1]]=(items,fn)=>items.map(fn);context[helpers[2]]=(items,fn)=>items.filter(fn);}
const dynamic=migrationSource.match(/some\(e=>([\w$]+)\(e\)\.some/);
if(dynamic)context[dynamic[1]]=spec=>{if(spec.type.startsWith('COMFY_DYNAMIC'))throw Error('Dynamic input groups are outside this static canvas oracle');return[spec];};
const controlSource=functionCode('addValueControlWidgets');
for(const match of controlSource.matchAll(/\b\w+\[([\w$]+)\]=!0/g))context[match[1]]=Symbol();
for(const match of controlSource.matchAll(/(updateControlWidgetLabel(?:\$\d+)?)\(/g))context[match[1]]=()=>{};
for(const match of controlSource.matchAll(/\b([\w$]+)\(t\)/g))context[match[1]]=()=>false;
for(const match of controlSource.matchAll(/values:\[\.\.\.([\w$]+)\]/g))context[match[1]]=['fixed','increment','decrement','randomize'];
const intSource=bodyAt('var useIntWidget=()=>');
const settings=intSource.match(/!([\w$]+)\(\)\.get/);
if(settings)context[settings[1]]=()=>({get:()=>false});
const transform=intSource.match(/void 0,void 0,([\w$]+)\(t\)/);
if(transform)context[transform[1]]=spec=>[spec.type,spec];
vm.runInContext(intSource,context);
if(request.seedCompatibility){
  context.app={registerExtension(extension){context.seedExtension=extension;}};
  vm.runInContext(fs.readFileSync(request.seedCompatibility,'utf8').replace(/^import .*;\s*/m,''),context);
}
const results=[];
for(const item of request.nodes){
  const info=request.infos[item.type];
  if(!info)throw Error('Unknown canvas node '+item.type);
  const specs={};
  for(const group of ['required','optional']){
    for(const [name,raw] of Object.entries(info.input[group]??{})){
      const options=raw[1]??{};
      const type=Array.isArray(raw[0])?'COMBO':raw[0];
      specs[name]={...options,name,type,isOptional:group==='optional'};
      if(Array.isArray(raw[0]))specs[name].options=raw[0];
    }
  }
  const ordered=context.getOrderedInputSpecs(info,specs);
  const node={widgets:[],inputs:[],addWidget(type,name,value,callback,options={}){
    const widget={type,name,value,callback,options,serialize:options.serialize};
    this.widgets.push(widget);return widget;
  }};
  const sockets=[];
  for(const spec of ordered){
    if(spec.forceInput||!['INT','FLOAT','BOOLEAN','STRING','COMBO'].includes(spec.type)){
      sockets.push({name:spec.name,type:spec.type,isOptional:spec.isOptional});continue;
    }
    if(spec.type==='INT')context.useIntWidget()(node,spec);
    else node.addWidget(spec.type.toLowerCase(),spec.name,spec.default??spec.options?.[0]??(spec.type==='BOOLEAN'?false:spec.type==='FLOAT'?0:''));
  }
  const restored=context.migrateWidgetsValues(specs,node.widgets,item.widgets_values);
  const shadow=context.computeLegacyWidgetShadow(node.widgets,restored);
  let compatibility;
  if(context.seedExtension){
    class FrontendNode{
      configure(info,...args){
        info.widgets_values=context.migrateWidgetsValues(specs,this.widgets,info.widgets_values??[]);
        for(const entry of context.computeLegacyWidgetShadow(this.widgets,info.widgets_values)){
          this.widgets.find(widget=>widget.name===entry.name).value=entry.value;
        }
        this.lastConfigured=info;
        return this.onConfigure(info,...args);
      }
      onConfigure(info,...args){
        this.previousConfigureArgs=[info,...args];
        if(item.priorChoice)this.widgets.find(widget=>widget.name==='control_after_generate').value=item.priorChoice;
        return 'prior-configure-result';
      }
      onSerialize(info,...args){
        this.previousSerializeArgs=[info,...args];
        info.properties={other_extension:'preserved'};
        return 'prior-serialize-result';
      }
    }
    await context.seedExtension.beforeRegisterNodeDef(FrontendNode,{...info,name:item.type});
    Object.setPrototypeOf(node,FrontendNode.prototype);
    const input=JSON.parse(JSON.stringify(item));
    const configuredReturn=node.configure(input,'configure-marker');
    const control=node.widgets.find(widget=>widget.name==='control_after_generate');
    const configuredControl=control?.value;
    const configuredValues=Object.fromEntries(node.widgets.filter(widget=>widget.serialize!==false).map(widget=>[widget.name,widget.value]));
    if(control&&item.roundtripChoice)control.value=item.roundtripChoice;
    const serialized={},serializedReturn=node.onSerialize(serialized,'serialize-marker');
    Object.assign(serialized,context.serialiseWidgetValues(node.widgets));
    if(control)control.value='randomize';
    const priorChoice=item.priorChoice;
    delete item.priorChoice;
    node.configure(serialized,'roundtrip-marker');
    item.priorChoice=priorChoice;
    compatibility={configuredControl,configuredValues,configuredReturn,serializedReturn,
      configureArgumentPreserved:node.previousConfigureArgs[1]==='roundtrip-marker',
      serializeArgumentPreserved:node.previousSerializeArgs[1]==='serialize-marker',
      roundtripControl:control?.value,properties:serialized.properties,
      inputValuesUnchanged:JSON.stringify(input.widgets_values)===JSON.stringify(item.widgets_values)};
  }
  results.push({id:item.id,type:item.type,ordered:ordered.map(spec=>spec.name),sockets,
    widgets:node.widgets.map(({name,serialize,value})=>({name,serialize:serialize!==false,default:value})),
    restored:Object.fromEntries(shadow.map(({name,value})=>[name,value])),restoredLength:restored.length,compatibility});
}
process.stdout.write(JSON.stringify({frontendSource:source.path,nodes:results}));
