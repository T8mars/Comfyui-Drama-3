/* Mutually exclusive individual casting and saved casts for the scene page. */
(() => {
'use strict';
const $ = id => document.getElementById(id);
const bindingKey = 'drama-basic-scene-voices-v1';
const modeKey = 'drama-basic-scene-voice-mode-v1';
function saved(key) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch { return {}; }
}
const bindings = saved(bindingKey), settings = saved(modeKey);
let mode = settings.mode === 'cast' ? 'cast' : 'individual';
let castId = typeof settings.cast_id === 'string' ? settings.cast_id : '';
let signature = '', active = false;
const panel = document.createElement('section');
panel.id = 'basic-scene-voices';
panel.hidden = true;
panel.innerHTML = `<h3>角色选声</h3>
  <div class="scene-voice-modes" role="tablist" aria-label="选声方式">
    <button id="scene-mode-individual" type="button" role="tab" aria-controls="scene-voice-individual">单独选角</button>
    <button id="scene-mode-cast" type="button" role="tab" aria-controls="scene-voice-group">选择选角组</button>
  </div>
  <div id="scene-voice-individual" role="tabpanel" aria-labelledby="scene-mode-individual">
    <p id="scene-voice-help" class="hint"></p><div id="scene-voice-rows"></div>
  </div>
  <div id="scene-voice-group" role="tabpanel" aria-labelledby="scene-mode-cast" hidden>
    <label>选角组<select id="scene-voice-cast" aria-label="选择已保存的选角"></select></label>
    <p id="scene-group-help" class="hint"></p><div id="scene-group-rows"></div>
  </div>
  <p id="scene-voice-status" class="hint" role="status"></p>`;
$('characters').closest('label').after(panel);
const sidebarTitle = document.querySelector('.voices-panel .section-title h2');
const sidebarDescription = document.querySelector('.voices-panel .section-title p');
const originalTitle = sidebarTitle.textContent, originalDescription = sidebarDescription.textContent;
const css = document.createElement('style');
css.textContent = `
#basic-scene-voices{padding:17px;margin:14px 0;border:1px solid #efd3e2;border-radius:14px;background:#fff9fc;min-width:0}
#basic-scene-voices [hidden],#basic-scene-voices[hidden]{display:none!important}
#basic-scene-voices h3{font-size:14px;margin:0 0 12px;color:#8c5674}
.scene-voice-modes{display:flex;padding:4px;gap:4px;background:#f8eaf1;border-radius:10px;margin-bottom:14px}
.scene-voice-modes button{flex:1;min-width:0;padding:10px 12px;border-radius:8px;background:transparent;color:#946e82;font-size:13px}
.scene-voice-modes button[aria-selected="true"]{background:white;color:#d7497c;font-weight:700;box-shadow:0 2px 8px #b46d8e12}
#basic-scene-voices .hint{margin:0 0 12px}
#scene-voice-group>label{margin-bottom:10px}
.scene-voice-row,.scene-group-row{padding:12px 0;border-top:1px solid #f1dfeb;display:grid;grid-template-columns:75px minmax(0,1fr) minmax(0,1fr);gap:9px;align-items:center;min-width:0}
.scene-voice-row strong,.scene-group-row strong{color:#9e5e80;font-size:13px;overflow-wrap:anywhere}
.scene-voice-row label{margin:0;display:grid;gap:4px;font-size:11px;font-weight:400;min-width:0}
.scene-voice-row select{width:100%;min-width:0;margin:0;font-size:12px}
.scene-voice-row audio,.scene-group-row audio{grid-column:2/-1;width:100%;min-width:0;height:33px}
.scene-group-description{grid-column:2/-1;color:#566579;font-size:12px;line-height:1.6;overflow-wrap:anywhere}
.scene-group-description small{display:block}
.scene-role-voice-mode #cd-legacy-voices,.scene-role-voice-mode #reference-list{display:none!important}
@media(max-width:750px){.scene-voice-row,.scene-group-row{grid-template-columns:minmax(0,1fr)}.scene-voice-row audio,.scene-group-row audio,.scene-group-description{grid-column:auto}}
`;
document.head.append(css);
function library() { return window.CreativeDesk?.state.library || {voices:[], casts:[]}; }
function selectedCast() { return (library().casts || []).find(cast => cast.id === castId); }
function roles() {
  const found = new Map();
  for (const raw of $('text').value.split(/\r?\n/)) {
    const tag = raw.match(/^\s*<\|speaker:(\d+)\|>/);
    if (tag) {
      const speaker = Number(tag[1]);
      found.set('i:' + speaker, {key:'i:' + speaker, name:'角色 ' + speaker, speaker});
      continue;
    }
    const match = raw.match(/^\s*(?:\[[^\[\]]*\]\s*)*([^<>:：\[\]]{1,48})[:：]/);
    if (match) {
      const name = match[1].trim();
      found.set('n:' + name, {key:'n:' + name, name, role:name});
    }
  }
  if (!found.size) for (const name of typeof sceneRoles === 'function' ? sceneRoles() : []) {
    found.set('n:' + name, {key:'n:' + name, name, role:name});
  }
  if (!found.size) found.set('i:0', {key:'i:0', name:'默认角色', speaker:0, unlabelled:true});
  return [...found.values()];
}
function persist() {
  try {
    localStorage.setItem(bindingKey, JSON.stringify(bindings));
    localStorage.setItem(modeKey, JSON.stringify({mode, cast_id:castId}));
  } catch {}
}
function castBinding(cast, role) {
  return role.role ? cast.bindings[role.role] : Object.values(cast.bindings).find(value => value.speaker === role.speaker);
}
function read() {
  const result = [], cast = selectedCast(), items = roles();
  if (mode === 'cast') {
    if (!cast) throw Error('请选择选角组，或切换到“单独选角”。');
    if (items.some(role => role.unlabelled) && Object.keys(cast.bindings).length > 1) {
      throw Error('此选角组包含多个角色。请在台词前写角色名，例如“女主：台词”，以便对应组内音色。');
    }
  }
  for (const role of items) {
    const value = mode === 'cast' ? castBinding(cast, role) : bindings[role.key];
    if (!value?.voice_id) {
      if (mode === 'cast') throw Error('选角组没有对应“' + role.name + '”的音色。请检查角色名，或切换到“单独选角”。');
      continue;
    }
    if (!value.version_id) throw Error('请为“' + role.name + '”选择固定音色版本。');
    result.push({voice_id:value.voice_id, version_id:value.version_id,
      ...(role.role ? {role:role.role} : {speaker:role.speaker})});
  }
  return result;
}
function fingerprint() {
  const lib = library();
  return JSON.stringify({mode, castId, items:roles(), bindings,
    voices:(lib.voices || []).map(voice => [voice.id, voice.name, voice.archived, voice.latest_version_id,
      voice.versions.map(version => [version.id, version.delivery, version.language])]),
    casts:(lib.casts || []).map(cast => [cast.id, cast.name, cast.bindings])});
}
function pauseReferences() { for (const audio of panel.querySelectorAll('audio')) audio.pause(); }
function referencePlayer(versionId, name) {
  const audio = document.createElement('audio');
  audio.controls = true; audio.preload = 'metadata'; audio.setAttribute('aria-label', name + '的参考试听');
  if (versionId) audio.src = '/api/creative/versions/' + encodeURIComponent(versionId) + '/audio';
  else audio.hidden = true;
  audio.onplay = () => { for (const other of document.querySelectorAll('audio')) if (other !== audio) other.pause(); };
  return audio;
}
function renderIndividuals() {
  const lib = library(), items = roles(), box = $('scene-voice-rows');
  box.replaceChildren();
  $('scene-voice-help').textContent = items.some(role => role.unlabelled)
    ? '从音色库选择单个音色并试听。多人对白按“导演：台词”“女主：台词”填写后，可为每人分别选声。'
    : '逐个为角色选择音色和固定版本，可直接试听；未绑定的角色由模型设计声音。';
  for (const role of items) {
    const remembered = bindings[role.key] || {}, row = document.createElement('div');
    row.className = 'scene-voice-row'; row.dataset.role = role.name;
    const title = document.createElement('strong'); title.textContent = role.name;
    const voice = document.createElement('select'); voice.className = 'scene-role-voice';
    voice.setAttribute('aria-label', role.name + '的音色'); voice.add(new Option('模型设计声音 · 无参考', ''));
    for (const item of lib.voices || []) if (item.versions.length && (!item.archived || item.id === remembered.voice_id)) {
      voice.add(new Option(item.name, item.id));
    }
    if (remembered.voice_id && ![...voice.options].some(item => item.value === remembered.voice_id)) {
      voice.add(new Option('原固定音色当前不可用', remembered.voice_id));
    }
    voice.value = remembered.voice_id || '';
    const version = document.createElement('select'); version.className = 'scene-role-version';
    version.setAttribute('aria-label', role.name + '的固定版本');
    const audio = referencePlayer('', role.name);
    function fill() {
      const item = (lib.voices || []).find(value => value.id === voice.value), chosen = bindings[role.key] || {};
      version.replaceChildren();
      for (const value of item?.versions || []) version.add(new Option(
        (value.delivery || '默认演出') + ' · ' + (value.language || '') + ' · ' + value.id.slice(0,6), value.id));
      if (!item && !chosen.version_id) version.add(new Option('无需参考版本', ''));
      version.disabled = !item || mode !== 'individual';
      if (chosen.version_id && ![...version.options].some(value => value.value === chosen.version_id)) {
        version.add(new Option('固定旧版本 · ' + chosen.version_id.slice(0,6), chosen.version_id));
      }
      version.value = chosen.version_id || item?.latest_version_id || '';
      audio.pause(); audio.hidden = !voice.value || !version.value;
      if (!audio.hidden) audio.src = '/api/creative/versions/' + encodeURIComponent(version.value) + '/audio';
      else { audio.removeAttribute('src'); audio.load(); }
    }
    voice.onchange = () => {
      const item = (library().voices || []).find(value => value.id === voice.value);
      bindings[role.key] = item ? {voice_id:item.id, version_id:item.latest_version_id} : {};
      persist(); fill(); signature = fingerprint();
      $('scene-voice-status').textContent = '当前使用单独选角，已更新“' + role.name + '”的音色。';
    };
    version.onchange = () => {
      bindings[role.key] = {voice_id:voice.value, version_id:version.value};
      persist(); fill(); signature = fingerprint();
    };
    const voiceLabel = document.createElement('label'); voiceLabel.textContent = '音色库'; voiceLabel.append(voice);
    const versionLabel = document.createElement('label'); versionLabel.textContent = '音色版本'; versionLabel.append(version);
    row.append(title, voiceLabel, versionLabel, audio); box.append(row); fill();
  }
}
function renderGroup() {
  const lib = library(), select = $('scene-voice-cast'), box = $('scene-group-rows');
  select.replaceChildren(new Option('请选择选角组', ''));
  for (const cast of lib.casts || []) select.add(new Option(cast.name, cast.id));
  select.value = castId; box.replaceChildren();
  const cast = selectedCast();
  if (!cast) { $('scene-group-help').textContent = '选择一组选角后，整组使用其中保存的音色与固定版本。'; return; }
  $('scene-group-help').textContent = '当前仅使用“' + cast.name + '”。台词请使用对应角色名：' + Object.keys(cast.bindings).join('、') + '。';
  for (const [role, value] of Object.entries(cast.bindings)) {
    const voice = (lib.voices || []).find(item => item.id === value.voice_id);
    const version = voice?.versions.find(item => item.id === value.version_id);
    const row = document.createElement('div'); row.className = 'scene-group-row'; row.dataset.role = role;
    const title = document.createElement('strong'); title.textContent = role;
    const description = document.createElement('div'); description.className = 'scene-group-description';
    description.textContent = voice?.name || value.voice_name || '固定音色';
    const detail = document.createElement('small');
    detail.textContent = (version?.delivery || value.delivery || '默认演出') + ' · 固定版本 ' + value.version_id.slice(0,6);
    description.append(detail); row.append(title, description, referencePlayer(value.version_id, role)); box.append(row);
  }
}
function render(force = false) {
  const on = typeof page !== 'undefined' && page === 'scene';
  document.body.classList.toggle('scene-role-voice-mode', on); panel.hidden = !on;
  sidebarTitle.textContent = on ? '参考音频' : originalTitle;
  sidebarDescription.textContent = on ? '上传新参考后，在中间的音色库中选择使用。' : originalDescription;
  if (active && !on) pauseReferences(); active = on;
  if (!on) return;
  const next = fingerprint(); if (!force && signature === next) return;
  signature = next; pauseReferences();
  for (const value of ['individual','cast']) {
    const button = $('scene-mode-' + value), selected = mode === value;
    button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
  }
  $('scene-voice-individual').hidden = mode !== 'individual'; $('scene-voice-group').hidden = mode !== 'cast';
  renderIndividuals(); renderGroup();
  for (const select of $('scene-voice-individual').querySelectorAll('select')) {
    if (mode !== 'individual') select.disabled = true;
  }
  $('scene-voice-cast').disabled = mode !== 'cast';
}
function setMode(value) {
  mode = value; persist(); $('scene-voice-status').textContent = ''; render(true);
}
for (const value of ['individual','cast']) {
  $('scene-mode-' + value).onclick = () => setMode(value);
  $('scene-mode-' + value).onkeydown = event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault(); setMode(event.key === 'Home' ? 'individual' : event.key === 'End' ? 'cast' : mode === 'cast' ? 'individual' : 'cast');
    $('scene-mode-' + mode).focus();
  };
}
$('scene-voice-cast').onchange = () => { castId = $('scene-voice-cast').value; persist(); render(true); };
const old = window.getCreativeVoiceVersions;
window.getCreativeVoiceVersions = () => {
  if (typeof page !== 'undefined' && page === 'scene') return read();
  return old?.() || [];
};
for (const id of ['text','characters']) $(id).addEventListener('input', () => render());
$('tabs').addEventListener('click', () => queueMicrotask(() => render()));
setInterval(() => { if (window.CreativeDesk?.state.initialized) render(); }, 650);
window.BasicSceneVoices = {roles, read, refresh:() => render(true),
  requestOptions:() => ({voice_selection_mode:mode, ...(mode === 'cast' ? {voice_cast_id:castId} : {})})};
})();
