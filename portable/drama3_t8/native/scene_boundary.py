"""Bounded recovery when a scene ends a line before its required separator.

The scene grammar writes a newline before an audio separator. Redirect one
premature EOS per line to that newline, then let the model sample the separator
and the next line normally. This never inserts a separator or any audio codes.
"""


class SceneBoundaryPolicy:
    def __init__(self, expected_segments, *, newline_id, eos_id, separator_id,
                 semantic_start, semantic_end, on_correction=None):
        if type(expected_segments) is not int or expected_segments < 1:
            raise ValueError("Scene segment count must be a positive integer")
        self.expected_segments=expected_segments
        self.newline_id=newline_id
        self.eos_id=eos_id
        self.separator_id=separator_id
        self.semantic_start=semantic_start
        self.semantic_end=semantic_end
        self.on_correction=on_correction
        self.completed=0
        self.audio_frames=0
        self.redirected=False

    def select(self, raw_token, step):
        if raw_token==self.eos_id:
            if not self.audio_frames:
                raise RuntimeError("Scene generation ended with an empty audio segment")
            if self.completed < self.expected_segments-1:
                if self.redirected:
                    raise RuntimeError("Scene generation ended again before producing the required separator")
                self.redirected=True
                if self.on_correction is not None:
                    self.on_correction(dict(step=step,line=self.completed+1,
                        raw_token_id=raw_token,emitted_token_id=self.newline_id,
                        reason="early_eos_before_scene_boundary"))
                return self.newline_id
        elif raw_token==self.separator_id:
            if not self.audio_frames or self.completed>=self.expected_segments-1:
                raise RuntimeError("Scene generation produced an empty or extra audio segment")
            self.completed+=1
            self.audio_frames=0
            self.redirected=False
        elif self.semantic_start <= raw_token <= self.semantic_end:
            self.audio_frames+=1
        return raw_token
