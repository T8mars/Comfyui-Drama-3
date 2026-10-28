// Adapted from PedroMarinhoDev/ComfyUI-Drama3 @ 8cb3c643 (GPL-3.0); see licenses/NOTICE-DRAMA3-EDITOR.
// The Drama 3 Editor node: the editor core (editor_core.js) on a ComfyUI node. The node keeps the editor's settings
// and edits in its hidden "edits" widget and the names of its last take's preview files in its properties; the server
// plans edits at /drama3_t8/editor/plan, and "Use these edits" queues the workflow, which renders the same plan.

import { EditorView, EditorWindow, TakePreview, injectStyle, loadTake } from "./editor_core.js";

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

function viewUrl(name, subfolder, type) {
  return api.apiURL(`/view?${new URLSearchParams({ filename: name, subfolder, type })}`);
}

function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
}

async function requestPlan(analysis, edits) {
  const response = await api.fetchApi("/drama3_t8/editor/plan", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ analysis, edits }),
  });
  if (!response.ok) { const data=await response.json().catch(()=>({})); throw new Error(data.error || `HTTP ${response.status}`); }
  return response.json();
}

app.registerExtension({
  name: "T8Drama3.Editor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "T8DramaEditor") return;
    injectStyle();
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      const editsW = this.widgets?.find((w) => w.name === "edits");
      hideWidget(editsW);
      const readEdits = () => { try { return JSON.parse(editsW?.value || "{}"); } catch { return {}; } };
      const writeEdits = (edits) => {
        if (!editsW) return;
        editsW.value = JSON.stringify(edits);
        editsW.callback?.(editsW.value);
        node.setDirtyCanvas?.(true, true);
      };
      const view = new EditorView({ readEdits, writeEdits, apply: () => app.queuePrompt(0, 1), plan: requestPlan });
      this.drama3View = view;
      let window_ = null;
      const preview = new TakePreview(view, {
        emptyText: "运行画布后在这里试听；支持分角色音量、剪切、停顿与字幕。",
        onOpen: () => { if (!window_) window_ = new EditorWindow(view, () => { window_ = null; view.emit(); }); },
      });
      for (const type of ["pointerdown", "wheel"]) preview.root.addEventListener(type, (e) => e.stopPropagation());
      const widget = this.addDOMWidget("drama3_editor", "div", preview.root, { serialize: false, getMinHeight: () => 190 });
      widget.serialize = false;
      if (!this.size || this.size[0] < 420 || this.size[1] < 300) this.setSize([Math.max(this.size?.[0] ?? 0, 460), Math.max(this.size?.[1] ?? 0, 320)]);

      let loaded = null;
      let loading = 0;
      const load = async (entry) => {
        if (loaded === entry.audio + entry.analysis) { await view.replan(); return; }
        const generation=++loading;
        preview.setStatus("正在载入音频…");
        try {
          // Workflow's hidden widget may contain newer, unrendered edits than
          // the last executed preview stored in properties. Preserve that draft.
          if (entry.edits && readEdits().source !== entry.source) writeEdits(entry.edits);
          const accepted=await loadTake(view, viewUrl(entry.analysis, entry.subfolder, entry.type), viewUrl(entry.audio, entry.subfolder, entry.type));
          if (!accepted || generation!==loading) return;
          loaded = entry.audio + entry.analysis;
          preview.setStatus(null);
          if (entry.edits_reset) preview.root.title='所选演出已改变，旧剪辑已重置；当前输出保持新演出的原始声音。';
          window_?.renderAll();
        } catch (err) {
          if (generation!==loading) return;
          preview.setStatus("预览文件已失效，请重新运行画布。已保存的剪辑保留。");
          console.warn("Drama 3 Editor:", err);
        }
      };

      const onExecuted = this.onExecuted;
      this.onExecuted = function (output) {
        onExecuted?.apply(this, arguments);
        const entry = output?.drama3_t8_editor?.[0];
        if (!entry) return;
        this.properties = this.properties || {};
        this.properties.drama3_editor = entry;
        load(entry);
      };
      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        hideWidget(editsW);
        const entry = this.properties?.drama3_editor;
        if (entry) load(entry);
      };
      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        onRemoved?.apply(this, arguments);
        loading++;
        window_?.close();
        preview.destroy();
        view.destroy();
      };
      return r;
    };
  },
});
