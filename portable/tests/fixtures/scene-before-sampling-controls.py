class T8DramaScene(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaScene", display_name="T8 Drama 3 · 多角色场景",
            category="T8/Drama3", inputs=[EngineType.Input("engine"), io.String.Input("script", multiline=True),
                io.String.Input("scene", default="", multiline=True),
                io.String.Input("characters", default="", multiline=True),
                ReferencesType.Input("references", optional=True),
                io.Int.Input("seed", default=42, min=0, max=CANVAS_SEED_MAX),
                io.Int.Input("max_frames", default=1200, min=1, max=8192), exact_seed_input(),
                io.Combo.Input('generation_mode',options=['together','per_line','per_line_context','per_speaker'],default='together',optional=True),
                io.Boolean.Input('split_long_text',default=False,optional=True),
                io.Int.Input('chunk_bytes',default=300,min=32,max=3000,optional=True)],
            outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, engine, script, scene="", characters="", references=None, seed=42, max_frames=1200, seed_text="",generation_mode='together',split_long_text=False,chunk_bytes=300):
        lines = [line for line in script.splitlines() if line.strip()]
        return io.NodeOutput(engine.scene(lines, scene, characters, tuple(references or ()),
                                         Sampling(seed=sampling_seed(seed, seed_text), max_frames=max_frames),generation_mode=generation_mode,
                                         split_long_text=split_long_text,chunk_bytes=chunk_bytes))
