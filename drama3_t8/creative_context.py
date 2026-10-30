"""Bounded, source-checked acoustic conditioning for creative workbench jobs.

Project cast IDs remain global. Only the isolated per-line context request maps
its active actor to local 0 and the immediately preceding other actor to local 1.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import re

_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")


def _role(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or "\x00" in value:
        raise ValueError("A creative role must be nonempty text")
    return value


def _actor(value):
    if type(value) is not int or not 0 <= value < 16:
        raise ValueError("Creative speaker must be an integer from 0 to 15")
    return value


def _line(raw):
    if isinstance(raw, str):
        return {"role": "旁白", "readout_text": raw}
    if not isinstance(raw, dict):
        raise ValueError("Creative lines must be text or objects")
    result = dict(raw)
    result["role"] = _role(raw.get("role", "旁白"))
    return result


def _text(line):
    for key in ("readout_override", "readout_text", "original_text", "text"):
        if line.get(key) is not None:
            if not isinstance(line[key], str):
                raise ValueError("Creative line text must be text")
            return line[key]
    return ""


def assign_speakers(cast, lines):
    """Return a copied cast with stable global IDs, including unreferenced roles.

    Existing explicit IDs (including intentional aliases) reserve their slots.
    New roles follow script first-use order; string lines are narrator lines.
    """
    if not isinstance(cast, dict) or not isinstance(lines, (list, tuple)):
        raise ValueError("Cast must be an object and lines must be a list")
    result = {}
    for name, binding in cast.items():
        _role(name)
        if binding is None:
            binding = {}
        if not isinstance(binding, dict):
            raise ValueError("Cast bindings must be objects")
        result[name] = copy.deepcopy(binding)
        if binding.get("speaker") is not None:
            _actor(binding["speaker"])
    normalized = [_line(line) for line in lines]
    order = list(dict.fromkeys([line["role"] for line in normalized] + list(result)))
    for line in normalized:
        name, text = line["role"], _text(line)
        for marker in re.finditer(r"<\|speaker", text):
            match = _TAG.match(text, marker.start())
            if match is None or str(int(match.group(1))) != match.group(1):
                raise ValueError("Creative speaker tags must use canonical integers from 0 to 15")
        numbers = {int(match.group(1)) for match in _TAG.finditer(text)}
        if len(numbers) > 1:
            raise ValueError("Split multiple speakers into separate creative lines")
        if numbers:
            selected = _actor(next(iter(numbers)))
            if result.get(name, {}).get("speaker") not in (None, selected):
                raise ValueError(f"Explicit speaker conflicts with the cast role {name}")
            result.setdefault(name, {})["speaker"] = selected
    occupied = {binding["speaker"] for binding in result.values() if binding.get("speaker") is not None}
    for name in order:
        binding = result.setdefault(name, {})
        if binding.get("speaker") is None:
            free = next((number for number in range(16) if number not in occupied), None)
            if free is None:
                raise ValueError("More than 16 unaliased creative speakers; assign intentional aliases explicitly")
            binding["speaker"] = free
            occupied.add(free)
    return result


def _descriptions(job):
    descriptions, seen = [], {}
    for candidate in job.get("context_candidates") or []:
        if not isinstance(candidate, dict) or candidate.get("status", "completed") != "completed":
            raise ValueError("Context candidates must be completed")
        snapshot = candidate.get("input_snapshot") or {}
        raw_lines = snapshot.get("lines")
        if raw_lines is None and isinstance(snapshot.get("line"), dict):
            raw_lines = [snapshot["line"]]
        if not isinstance(raw_lines, list) or not raw_lines:
            raise ValueError("Context candidate has no recorded script lines")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Context candidate is missing its ID")
        description = {"candidate": candidate, "id": candidate_id, "lines": [_line(line) for line in raw_lines],
                       "cast": snapshot.get("cast") or {}, "snapshot": snapshot}
        identity = (candidate.get("session_path"), candidate.get("conditioning_session_path"))
        if candidate_id in seen:
            if seen[candidate_id] != identity:
                raise ValueError("Conflicting context candidate identity")
            continue
        seen[candidate_id] = identity
        descriptions.append(description)
    return descriptions


def _scope(job, descriptions):
    cast, explicit = {}, set()
    def merge(incoming):
        if not isinstance(incoming, dict):
            raise ValueError("Context cast must be an object")
        for name, binding in incoming.items():
            _role(name)
            if binding is None:
                binding = {}
            if not isinstance(binding, dict):
                raise ValueError("Context cast binding must be an object")
            if binding.get("speaker") is not None:
                actor = _actor(binding["speaker"])
                if cast.get(name, {}).get("speaker") not in (None, actor):
                    raise ValueError(f"Historical global speaker conflicts for role {name}")
                explicit.add(name)
            cast[name] = {**cast.get(name, {}), **{key: value for key, value in binding.items() if value is not None}}
    speaker_map = job.get("speaker_map") or job.get("input_snapshot", {}).get("speaker_map") or {}
    merge({name: {"speaker": number} for name, number in speaker_map.items()})
    merge(job.get("speaker_cast") or {})
    for description in descriptions:
        merge(description["cast"])
    merge(job.get("cast") or {})
    current = job.get("lines") or ([job["line"]] if isinstance(job.get("line"), dict) else [])
    current = [_line(line) for line in current]
    if not current:
        raise ValueError("A creative job needs its actual current lines")
    lookahead = [_line(line) for line in job.get("lookahead_lines") or []]
    all_lines = [line for item in descriptions for line in item["lines"]] + current + lookahead
    assigned = assign_speakers(cast, all_lines)
    return assigned, current, lookahead, explicit


def _load_session(path):
    from .state import load_session
    return load_session(path)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(description, job):
    candidate = description["candidate"]
    conditioning = bool(candidate.get("conditioning_session_path"))
    key = "conditioning_session_path" if conditioning else "session_path"
    path = candidate.get(key)
    if not isinstance(path, (str, Path)) or not Path(path).is_absolute():
        raise ValueError("Context Session must have a resolved local file path")
    path = Path(path)
    accepted = {"conditioning", "conditioning_session"} if conditioning else {"session"}
    records = [record for record in candidate.get("artifacts", []) if record.get("kind") in accepted]
    if len(records) > 1:
        raise ValueError("Ambiguous context Session artifacts")
    record = records[0] if records else {}
    checksum = record.get("sha256") or candidate.get("conditioning_session_sha256" if conditioning else "session_sha256")
    if not checksum:
        dependencies = job.get("input_snapshot", {}).get("context", [])
        checksum = next((item.get("session_sha256") for item in dependencies if item.get("candidate_id") == description["id"]), None)
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        raise ValueError("Context Session needs its recorded SHA256")
    if not path.is_file() or (record.get("bytes") is not None and path.stat().st_size != record["bytes"]):
        raise ValueError("Context Session artifact is missing or changed")
    return path, checksum.lower(), conditioning


def _load(description, job):
    from .state import Session
    path, checksum, conditioning = _artifact(description, job)
    if _hash(path) != checksum:
        raise ValueError("Context Session checksum mismatch")
    try:
        archive = _load_session(path)
    except Exception as exc:
        raise ValueError("Context Session cannot be loaded") from exc
    if _hash(path) != checksum:
        raise ValueError("Context Session changed during loading")
    if not isinstance(archive, Session) or not archive.takes:
        raise ValueError("Context Session has no completed Take")
    root = archive.takes[-1]
    return {"description": description, "session": Session(archive.references, (root,), archive.direction),
            "root": root, "sha256": checksum, "conditioning": conditioning, "histories": None}


def _actors(description, assigned):
    return [assigned[line["role"]]["speaker"] for line in description["lines"]]


def _source(data, actor, assigned, *, first=False):
    """Extract an actual one-actor acoustic source, never relabel a mixed scene."""
    from .dialogue import own_history
    root, description = data["root"], data["description"]
    actors = _actors(description, assigned)
    unique = set(actors)
    structured = ("isolated_sources" in root.metadata or "scene_chunking" in root.metadata
                  or root.metadata.get("task") == "scene" or "render_infill" in root.metadata)
    if data["conditioning"]:
        if len(unique) != 1 or actor not in unique:
            raise ValueError("A conditioning Session must represent one recorded creative actor")
        recorded = root.metadata.get("creative_global_speaker")
        if recorded is not None and _actor(recorded) != actor:
            raise ValueError("Conditioning Take global actor differs from its candidate snapshot")
        if root.metadata.get("history_eligible") is False or "render_infill" in root.metadata:
            raise ValueError("Conditioning Session must contain its actual complete encoded PCM")
        return root, "conditioning"
    if "render_infill" in root.metadata and len(unique) == 1:
        raise ValueError("Edited single-line audio needs its actual conditioning Session")
    if not structured:
        if len(unique) != 1 or actor not in unique or len({int(match.group(1)) for match in _TAG.finditer(root.prompt_text)}) > 1:
            raise ValueError("Cannot relabel mixed audio as one creative actor")
        recorded = root.metadata.get("creative_global_speaker")
        if recorded is not None and _actor(recorded) != actor:
            raise ValueError("Context Take global actor differs from its candidate snapshot")
        if root.metadata.get("history_eligible") is False:
            raise ValueError("Incomplete raw audio is not eligible for history")
        return root, "native"
    if data["histories"] is None:
        data["histories"] = own_history(data["session"])
    histories = data["histories"]
    names = [line["role"] for line in description["lines"] if assigned[line["role"]]["speaker"] == actor]
    original = None
    mapping = root.metadata.get("scene_speaker_map") or {}
    known = {mapping[name] for name in names if name in mapping}
    if len(known) == 1:
        original = next(iter(known))
    recorded_lines = root.metadata.get("line_speakers") or []
    if not recorded_lines and len(root.metadata.get("script_lines", [])) == len(actors):
        from .scene_cast import prepare_scene_cast
        from .dialogue import speaker
        rendered, _, _ = prepare_scene_cast(root.metadata["script_lines"], data["session"].references,
                                             data["session"].takes, force=True)
        recorded_lines = [speaker(line) for line in rendered]
    if original is None and len(recorded_lines) == len(actors):
        known = {number for number, selected in zip(recorded_lines, actors) if selected == actor}
        if len(known) == 1:
            original = next(iter(known))
    if original is None and actor in histories:
        original = actor
    if original is None and len(unique) == 1 and actor in unique and len(histories) == 1:
        original = next(iter(histories))
    turns = histories.get(original, [])
    if not turns:
        raise ValueError("No unambiguous single-actor source exists in this archived scene")
    return (turns[0] if first else turns[-1]), ("render_parent" if "render_infill" in root.metadata else "scene_source")


def _view(take, actor, local, *, preserve_stream=True):
    from .dialogue import recast
    if take.metadata.get("history_eligible") is False:
        raise ValueError("Incomplete raw audio cannot become creative history")
    metadata = {**take.metadata, "creative_global_speaker": actor, "creative_local_speaker": local}
    if not preserve_stream and metadata.get("task") == "text_stream":
        metadata["task"] = "tts"
    return replace(take, prompt_text=recast(take.prompt_text, local), metadata=metadata)


def condition_creative_context(job, base_session):
    """Return the bounded real prompt Session and serializable provenance."""
    from .state import Reference, Session
    from .dialogue import speaker, recast
    if not isinstance(job, dict) or not isinstance(base_session, Session):
        raise ValueError("Creative conditioning needs a job and native Session")
    descriptions = _descriptions(job)
    assigned, current, future, explicit = _scope(job, descriptions)
    mode, role = job.get("mode", "per_line"), current[0]["role"]
    actor = assigned[role]["speaker"]
    context = {"version": 1, "mode": mode, "role": role, "global_speaker": actor,
               "history": [], "candidate_ids": [], "lookahead_text": ""}
    metadata = {"creative_context": context}
    if len({assigned[line["role"]]["speaker"] for line in current}) == 1:
        metadata.update(creative_global_speaker=actor, creative_local_speaker=actor)
    loaded = {}
    def load(description):
        if description["id"] not in loaded:
            if len(loaded) >= 3:
                raise ValueError("Creative conditioning is limited to three candidate Sessions")
            loaded[description["id"]] = _load(description, job)
        return loaded[description["id"]]
    seen_turns = set()
    def turn(description, selected_actor, local, purpose, first=False, preserve_stream=True):
        data = load(description)
        source, kind = _source(data, selected_actor, assigned, first=first)
        identity = (description["id"], source.take_id, local)
        if identity in seen_turns:
            return None
        seen_turns.add(identity)
        view = _view(source, selected_actor, local, preserve_stream=preserve_stream)
        context["history"].append({"candidate_id": description["id"], "source_take_id": data["root"].take_id,
            "history_take_id": source.take_id, "global_speaker": selected_actor, "local_speaker": local,
            "purpose": purpose, "source": kind, "session_sha256": data["sha256"]})
        if description["id"] not in context["candidate_ids"]:
            context["candidate_ids"].append(description["id"])
        return view

    if mode == "per_line_context":
        if len(current) != 1:
            raise ValueError("Isolated creative context needs exactly one current line")
        own = [item for item in descriptions if actor in _actors(item, assigned)]
        history = []
        if own:
            history.append(turn(own[0], actor, 0, "own_first", first=True, preserve_stream=False))
            history.append(turn(own[-1], actor, 0, "own_latest", preserve_stream=False))
        previous = descriptions[-1] if descriptions else None
        previous_actor = _actors(previous, assigned)[-1] if previous else None
        if previous is not None and previous_actor != actor:
            history.append(turn(previous, previous_actor, 1, "previous_other", preserve_stream=False))
        else:
            previous_actor = None
        references = tuple(replace(reference, transcript=recast(reference.transcript, 0))
                           for reference in base_session.references if speaker(reference.transcript) == actor)
        # Single-utterance TTS uses the explicitly requested delivery, as the
        # independent/long-form/streaming modes do. Complete scene descriptions
        # belong to native Scene generation. Recasting already enforces the
        # real reference/history voice IDs without an invented system script.
        direction = base_session.direction
        if job.get("note"):
            direction = "\n\n".join(value for value in
                (direction, "Director revision:\n" + job["note"]) if value)
        context["direction_scope"] = "utterance_v1"
        context.update(local_speaker=0, previous_global_speaker=previous_actor, history_limit=3)
        metadata.update(creative_global_speaker=actor, creative_local_speaker=0)
        return Session(references, tuple(take for take in history if take is not None), direction), metadata

    if mode == "long_form" and descriptions:
        if base_session.references:
            context["policy"] = "explicit_references"
            metadata["long_context"] = {"version": 1, "policy": "explicit_references", "source_take_id": None,
                                       "completed_history_turns": 0}
            return replace(base_session, takes=()), metadata
        anchor = descriptions[0]
        anchor_actor = _actors(anchor, assigned)[0]
        data = load(anchor)
        source, kind = _source(data, anchor_actor, assigned, first=True)
        context.update(policy="first_new_take_reference", candidate_ids=[anchor["id"]], anchor={
            "candidate_id": anchor["id"], "source_take_id": data["root"].take_id, "history_take_id": source.take_id,
            "source": kind, "session_sha256": data["sha256"]})
        metadata["long_context"] = {"version": 1, "policy": "first_new_take_reference", "source_take_id": data["root"].take_id,
                                   "completed_history_turns": 0}
        reference = Reference(recast(source.prompt_text, actor), source.codes, "首段音色锚")
        return replace(base_session, references=(reference,), takes=()), metadata

    if mode == "text_stream":
        history = []
        if descriptions:
            first, last = descriptions[0], descriptions[-1]
            history.append(turn(first, _actors(first, assigned)[0], _actors(first, assigned)[0], "first_turn", first=True))
            history.append(turn(last, _actors(last, assigned)[-1], _actors(last, assigned)[-1], "latest_turn"))
        else:
            history = list(base_session.takes[:1])
            if len(base_session.takes) > 1:
                history.append(base_session.takes[-1])
        look = []
        for line in future:
            name = line["role"]
            if name not in explicit and name != role:
                raise ValueError("Lookahead role needs its assigned global speaker_map entry")
            look.append(recast(_text(line), assigned[name]["speaker"]))
        context.update(local_speaker=actor, history_limit=2, lookahead_text="\n".join(look))
        return replace(base_session, takes=tuple(take for take in history if take is not None)), metadata

    # Other modes retain their host's existing generation semantics. Imported
    # initial history is bounded without reconstructing or relabeling mixed audio.
    takes = base_session.takes if len(base_session.takes) <= 2 else (base_session.takes[0], base_session.takes[-1])
    return replace(base_session, takes=takes), metadata

