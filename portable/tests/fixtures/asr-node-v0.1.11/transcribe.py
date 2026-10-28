class T8DramaTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from .qwen3_asr.asr import LANGUAGES
        return io.Schema(node_id="T8DramaTranscribe", display_name="T8 Drama 3 · Qwen3-ASR 转录",
            category="T8/Drama3", inputs=[
                io.Combo.Input("asr_model", options=[name for name in folder_paths.get_filename_list("asr")
                    if name.lower().endswith(".safetensors")] or ["<custom path>"]),
                io.Audio.Input("audio"), io.Combo.Input("language", options=["auto", *LANGUAGES]),
                io.String.Input("context", default="", multiline=True),
                io.String.Input("model_path", default="", optional=True)],
            outputs=[io.String.Output(), io.String.Output()], is_output_node=True)

    @classmethod
    def execute(cls, asr_model, audio, language="auto", context="", model_path=""):
        from .qwen3_asr.asr import load_qwen3_asr, transcribe
        waveform = audio_waveform(audio["waveform"], audio["sample_rate"])
        path = (resolve_model_path(model_path, folder_paths.models_dir) if model_path.strip()
                else folder_paths.get_full_path_or_raise("asr", asr_model))
        model = load_qwen3_asr(str(path))
        text, found = transcribe(model, waveform, audio["sample_rate"],
                                 None if language == "auto" else language, context)
        return io.NodeOutput(text, found,ui=ui.PreviewText(text))
