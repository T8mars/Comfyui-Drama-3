"""Autoregressive frame generation: prefill the prompt rows, then per frame sample the main token from the audio
head, the other codes from the tower, and run the backbone on the new frame."""
import contextlib
import logging

import torch

import comfy.model_management
import comfy.ops
from .config import MIN_TEMPERATURE, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, RAS_WINDOW
from .scene_boundary import SceneBoundaryPolicy

# A take that falls silent without ending repeats a handful of codes: 96 frames (4.5 s) of speech hold 80-95
# distinct first codes and the most expressive measured stretch (sighs, laughs) 41; silent runaways 5-20.
STALL_WINDOW = 96
STALL_DISTINCT = 24
STALL_TAIL = 10


def sample_top_k_top_p(logits, temperature, top_k, top_p, generator, *, check_finite=True):
    """One id per row; drops the token crossing top_p except the highest-ranked one (the vendor's rule)."""
    if check_finite and not torch.isfinite(logits).all().item():
        raise FloatingPointError("Drama 3 produced non-finite sampling logits")
    values, ids = torch.sort(logits.float() / max(temperature, MIN_TEMPERATURE), descending=True, dim=-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= (top_k or values.shape[-1]))
    dropped[..., 0] = False
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    noise = torch.empty_like(probabilities).exponential_(generator=generator)
    return ids.gather(-1, (probabilities / noise).argmax(-1, keepdim=True)).squeeze(-1)


def embed_rows(embed, ids, dtype):
    """Token embeddings; an evicted dynamic table gathers the rows from its host copy instead of streaming whole."""
    if (hasattr(embed, "_v") and embed.weight_lowvram_function is None and len(embed.weight_function) == 0
            and comfy.ops.vbar_above_watermark(embed)):
        return embed.host_rows(ids, out_dtype=dtype)
    return embed(ids, out_dtype=dtype)


def generate(drama3, prompt, max_frames, temperature, top_k, top_p, seed, on_frame=None,
             on_audio=None, stop_token_ids=(), progress=None, on_finish=None,
             inspect_logits=None, head_override=None, extra_patchers=(),
             expected_segments=None, on_boundary=None):
    """Frames [frames, 1 + N] (CPU int64) sampled after the prompt rows [rows, 1 + N], without the end frame."""
    config = drama3.config
    backbone = drama3.model.model
    dtype = drama3.model.dtype
    layers, devices, norm, audio_head, tower = drama3.parts()
    output_ids = config.output_token_ids
    if head_override is not None:
        audio_head, output_ids = head_override
    first, last = devices[0], devices[-1]
    rows = prompt.to(last)
    prompt_rows = len(rows)
    capacity = prompt_rows + max_frames + 1
    if max_frames < 1 or capacity > config.text.max_position_embeddings:
        raise ValueError("The requested generation does not fit the model context")
    factory = getattr(backbone, "init_layer_caches", None)
    caches = (factory(capacity, layers, devices, dtype) if factory is not None
              else backbone.init_caches(capacity, devices, dtype))
    tower_caches = tower.init_caches(capacity, last, dtype)
    generator = torch.Generator(device=last).manual_seed(seed)
    candidates = torch.tensor(output_ids, device=last)
    separator_id = drama3.tokenizer.token_to_id("<|audio_separator|>")
    scene_boundary=None
    if expected_segments is not None:
        newline=drama3.tokenizer.encode("\n").ids
        if len(newline)!=1:
            raise ValueError("Scene boundary recovery requires a single newline token")
        scene_boundary=SceneBoundaryPolicy(expected_segments,newline_id=newline[0],
            eos_id=config.eos_token_id,separator_id=separator_id,
            semantic_start=config.semantic_start_token_id,semantic_end=config.semantic_end_token_id,
            on_correction=on_boundary)

    backbone_graphs=None
    tower_graph=None
    graph_reason=None

    def run(frames):
        text = embed_rows(backbone.embed_tokens, frames[:, 0].to(first), dtype)
        service={}
        if len(frames)==1 and getattr(drama3.experts,'cpu_side',None) is not None:
            service['after_layer']=drama3.experts.after_layer
        if backbone_graphs is not None and len(frames)==1:
            x = backbone.run_layers(tower.embed_backbone(frames, text)[None], caches, layers, devices,graphs=backbone_graphs,**service)
        else:
            x = backbone.run_layers(tower.embed_backbone(frames, text)[None], caches, layers, devices,**service)
        with comfy.model_management.cuda_device_context(last):
            return norm(x)

    finite_residual_heads = []

    def sample(logits):
        # Tower heads depend on each preceding code. Accumulate on-device
        # validity flags and synchronize once before publishing the frame.
        finite_residual_heads.append(torch.isfinite(logits).all())
        return sample_top_k_top_p(logits, temperature, top_k, top_p, generator, check_finite=False)

    frames, codes = [], []
    finish_reason = "length"
    if hasattr(drama3,'session'):
        session=drama3.session(cpu_decode=first==last,extra_patchers=extra_patchers)
    else:
        session=drama3.experts.session(first, [*drama3.patchers(), *extra_patchers]) if drama3.experts is not None else contextlib.nullcontext()
    with session:
        cpu_decode=getattr(drama3.experts,'cpu_side',None) is not None
        if getattr(drama3,'cuda_graph',False):
            if first!=last or last.type!='cuda':graph_reason='CUDA graphs require one CUDA device'
            elif not getattr(drama3.experts,'sync_free',False):graph_reason='Expert backend is not synchronization-free'
            elif any(getattr(p,'is_dynamic',lambda:False)() for p in drama3.patchers()):
                graph_reason='Dynamic host weights use their own graph lifecycle'
            else:
                from .graphs import BackboneGraphs,TowerGraph,norm_residency_reason
                graph_reason=norm_residency_reason(layers,devices,tower,last)
                if graph_reason is None:
                    backbone_graphs=BackboneGraphs(drama3.experts)
                    tower_graph=TowerGraph(tower,tower_caches,sample,finite_residual_heads,generator,last)
        hidden = run(rows)
        history = torch.full((RAS_WINDOW,), -1, dtype=torch.long, device=last)
        window = rows[-RAS_WINDOW:, 0]
        history[RAS_WINDOW - len(window):] = window
        frame = rows[-1:]
        with comfy.model_management.cuda_device_context(last):
            if prompt_rows > 1:
                count = prompt_rows - 1
                tower.run(hidden[:, :-1], rows[1:], rows[:-1], tower.context(torch.arange(count, device=last), count, last, dtype), tower_caches)
            hidden = hidden[:, -1:]
            for step in range(max_frames):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if progress is not None:
                    progress(step, max_frames)
                position = prompt_rows - 1 + step
                logits = audio_head(hidden[0])
                if inspect_logits is not None:
                    inspect_logits(step, hidden[0], logits)
                if not torch.isfinite(logits).all().item():
                    raise FloatingPointError("Drama 3 produced non-finite logits")
                draw = candidates[sample_top_k_top_p(logits, temperature, top_k, top_p, generator, check_finite=False)]
                retry = candidates[sample_top_k_top_p(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, generator, check_finite=False)]
                repeated = (history == draw[:, None]).any(-1) & tower.is_semantic(draw)
                main = torch.where(repeated, retry, draw)
                main_id = main.item()
                if scene_boundary is not None:
                    selected=scene_boundary.select(main_id,step)
                    if selected!=main_id:
                        main_id=selected
                        main=torch.full_like(main,selected)
                if main_id == config.eos_token_id or main_id in stop_token_ids:
                    finish_reason = "eos" if main_id == config.eos_token_id else f"control:{main_id}"
                    break
                context = tower.context(torch.tensor([position], device=last), 0, last, dtype)
                if config.semantic_start_token_id <= main_id <= config.semantic_end_token_id:
                    finite_residual_heads.clear()
                    if tower_graph is not None:
                        frame,finite=tower_graph.step(main,hidden,frame,context)
                    else:
                        frame = tower.step(main, hidden, frame, context, tower_caches, sample)
                        finite=torch.stack(finite_residual_heads).all() if finite_residual_heads else torch.ones((),dtype=torch.bool,device=last)
                    if not finite.item():
                        raise FloatingPointError("Drama 3 produced non-finite residual code logits")
                    codes.append(frame[0, 1].item())
                    if on_audio is not None:
                        on_audio(frame[0, 1:].detach().cpu())
                else:
                    # Non-audio controls still advance the temporal caches, but have no sampled codebooks.
                    next_frame = torch.zeros((1, rows.shape[1]), device=last, dtype=torch.long)
                    next_frame[0, 0] = main
                    tower.run(hidden, next_frame, frame, context, tower_caches)
                    frame = next_frame
                frames.append(frame)
                if len(codes) >= STALL_WINDOW and len(set(codes[-STALL_WINDOW:])) <= STALL_DISTINCT:
                    logging.warning("Drama 3: the take went silent without ending after %.1f s; its silent tail was cut. "
                                    "Another seed or a voice reference usually fixes this.", len(frames) / 21.5)
                    del frames[len(frames) - STALL_WINDOW + STALL_TAIL:]
                    finish_reason = "stalled"
                    break
                history = torch.cat((history[1:], frame[:, 0]))
                if cpu_decode:drama3.experts.between_steps()
                if on_frame is not None:
                    on_frame(step)
                if step + 1 < max_frames:
                    hidden = run(frame)
    if on_finish is not None:
        on_finish(finish_reason)
    stats=backbone_graphs.report() if backbone_graphs is not None else dict(captures=0,replays=0,fallback_reasons=[])
    stats.update(tower_captures=tower_graph.runner.captures if tower_graph else 0,
        tower_replays=tower_graph.runner.replays if tower_graph else 0,disabled_reason=graph_reason,cpu_decode=cpu_decode)
    if tower_graph and tower_graph.runner.fallback_reason:stats['fallback_reasons'].append(tower_graph.runner.fallback_reason)
    drama3.last_graph_stats=stats
    if not frames:
        return torch.zeros((0, rows.shape[1]), dtype=torch.long)
    return torch.cat(frames).cpu()
