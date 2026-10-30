"""Bounded local diagnostics and quality-gated, callback-driven benchmarks.

This module never loads a model, starts a service, changes precision, synchronizes
CUDA, or resets the host's allocator counters. Engine integration supplies the
actual work and its phase/backend observations. Reports retain JSON, not tensors.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import threading
import time
from typing import Any, Mapping

SCHEMA = "drama3.performance.v1"
BENCHMARK_SCHEMA = "drama3.benchmark.v1"
PROFILE_SCHEMA = "drama3.performance-profile.v1"
REQUIRED_COVERAGE = frozenset({"short", "long", "reference", "multi_speaker"})
SETTINGS = frozenset({"gdn_backend", "moe_backend", "cuda_graph", "expert_cache_gb", "expert_placement"})
_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PEAKS = ("process_rss_bytes", "process_private_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes")


def _json(value):
    """Copy supported values and refuse tensors/objects rather than retaining them."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Performance values must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(type(k) is str for k in value):
            raise TypeError("Performance object keys must be strings")
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        return [_json(v) for v in value]
    raise TypeError(f"Performance data must be JSON values, not {type(value).__name__}")


def _encoded(value):
    return json.dumps(_json(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _positive(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be a finite {'nonnegative' if zero else 'positive'} number")
    return float(value)


class ResourceProbe:
    """Read own-process RAM and already-initialized Torch allocator counters.

    No Torch import or CUDA initialization occurs. CUDA values cover this
    process's allocator, not memory occupied by other desktop applications.
    Exact device-wide/allocator epoch peaks can instead be supplied by the
    caller's probe, with an explicit ``measurement_scope`` description.
    """
    def __init__(self, device=0):
        self.device = device

    def __call__(self):
        result = {"measurement_scope": "sampled own-process RSS/private and initialized Torch allocator current bytes"}
        try:
            import psutil
            memory = psutil.Process().memory_info()
            result["process_rss_bytes"] = int(memory.rss)
            if hasattr(memory, "private"):
                result["process_private_bytes"] = int(memory.private)
            result["system_available_bytes"] = int(psutil.virtual_memory().available)
        except (ImportError, OSError, RuntimeError) as exc:
            result["RAM_probe_error"] = type(exc).__name__
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            result["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(self.device))
            result["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(self.device))
        return result


class PerformanceRecorder:
    """One job's bounded, thread-safe timing and resource observations."""
    def __init__(self, request=None, binding=None, *, probe=None, clock=None,
                 sample_interval=.1, max_spans=256, max_backends=64):
        self.request = _json(request or {})
        self.binding = _json(binding or {})
        self.probe = probe or ResourceProbe()
        self.clock = clock or time.perf_counter
        self.sample_interval = _positive(sample_interval, "sample_interval", zero=True)
        if type(max_spans) is not int or not 1 <= max_spans <= 10000:
            raise ValueError("max_spans must be between1 and10000")
        if type(max_backends) is not int or not 1 <= max_backends <= 10000:
            raise ValueError("max_backends must be between1 and10000")
        self.max_spans, self.max_backends = max_spans, max_backends
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.started = self.ended = None
        self.spans, self.active, self.backends, self.marks = [], {}, [], {}
        self.dropped_spans = self.dropped_backends = self.samples = self.probe_errors = 0
        self.peaks, self.first_sample, self.last_sample = {}, None, None
        self.minimum_available = None
        self.status, self.error_type = "pending", None

    def _elapsed(self):
        now = self.clock()
        elapsed = now - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Performance clock must be monotonic and finite")
        return elapsed

    def start(self):
        with self.lock:
            if self.started is not None:
                return self
            self.started = self.clock()
            if type(self.started) not in (int,float) or not math.isfinite(self.started):
                raise ValueError("Performance clock must be finite")
            self.status = "running"
        self.sample()
        if self.sample_interval:
            self.thread = threading.Thread(target=self._sample_loop, daemon=True,
                                           name="drama3-performance")
            self.thread.start()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, error, tb):
        status = "ok" if kind is None else ("cancelled" if isinstance(error, BenchmarkCancelled) else "error")
        self.finish(status, error_type=None if kind is None else kind.__name__)
        return False

    def _sample_loop(self):
        while not self.stop.wait(self.sample_interval):
            self.sample()

    def sample(self):
        try:
            row = _json(self.probe())
            if not isinstance(row, dict):
                raise TypeError("Resource probe must return an object")
            for key in (*_PEAKS, "system_available_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                value = row.get(key)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("Resource bytes must be nonnegative integers")
        except Exception:
            with self.lock:
                self.probe_errors += 1
            return
        with self.lock:
            self.samples += 1
            if self.first_sample is None:
                self.first_sample = row
            self.last_sample = row
            for key in (*_PEAKS, "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                value = row.get(key)
                if value is not None:
                    self.peaks[key] = max(self.peaks.get(key, 0), value)
            available = row.get("system_available_bytes")
            if available is not None:
                self.minimum_available = available if self.minimum_available is None else min(self.minimum_available, available)

    @contextmanager
    def phase(self, name, *, cache_state="unknown", **metadata):
        if not _NAME.fullmatch(name) or cache_state not in {"cold", "warm", "unknown", "not_used"}:
            raise ValueError("Invalid performance phase or cache state")
        self.start()
        row = dict(name=name, cache_state=cache_state, metadata=_json(metadata),
                   start_seconds=self._elapsed(), status="running")
        with self.lock:
            if self.ended is not None:
                raise RuntimeError("Performance recorder is finished")
            token = id(row)
            self.active[token] = row
        self.sample()
        try:
            yield row
        except BaseException as exc:
            row["status"], row["error_type"] = "error", type(exc).__name__
            raise
        finally:
            with self.lock:
                if token in self.active:
                    self.active.pop(token)
                    row["end_seconds"] = self._elapsed()
                    row["elapsed_seconds"] = row["end_seconds"] - row["start_seconds"]
                    if row["status"] == "running":row["status"] = "ok"
                    if len(self.spans) < self.max_spans:self.spans.append(_json(row))
                    else:self.dropped_spans += 1
            self.sample()

    def mark(self, name, **metadata):
        if not _NAME.fullmatch(name):
            raise ValueError("Invalid performance mark")
        self.start()
        with self.lock:
            if self.ended is not None:
                raise RuntimeError("Performance recorder is finished")
            if name in self.marks:return False
            if len(self.marks) >= self.max_spans:raise ValueError("Too many performance marks")
            self.marks[name] = dict(seconds=self._elapsed(), metadata=_json(metadata))
        self.sample()
        return True

    def mark_first_codes(self, count=1):
        _positive(count, "code count", zero=True)
        return False if count == 0 else self.mark("first_codes", count=count)

    def mark_first_pcm(self, samples=None, sample_rate=None):
        if samples is not None:_positive(samples, "PCM samples", zero=True)
        if sample_rate is not None:_positive(sample_rate, "sample rate")
        return False if samples == 0 else self.mark("first_pcm", samples=samples, sample_rate=sample_rate)

    def record_backend(self, component, requested, actual, fallback_reason=None):
        if not _NAME.fullmatch(component):raise ValueError("Invalid backend component")
        self.start()
        row = dict(component=component, requested=_json(requested), actual=_json(actual),
                   fallback_reason=_json(fallback_reason), seconds=self._elapsed())
        with self.lock:
            if self.ended is not None:raise RuntimeError("Performance recorder is finished")
            if len(self.backends) < self.max_backends:self.backends.append(row)
            else:self.dropped_backends += 1

    def finish(self, status="ok", *, error_type=None):
        if status not in {"ok", "error", "cancelled", "unavailable"}:raise ValueError("Invalid performance status")
        self.start()
        self.stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
        with self.lock:
            if self.ended is not None:return self.report()
            self.ended = self._elapsed()
            self.status, self.error_type = status, error_type
            for row in self.active.values():
                row.update(status="unfinished", end_seconds=self.ended,
                           elapsed_seconds=self.ended-row["start_seconds"])
                if len(self.spans) < self.max_spans:self.spans.append(_json(row))
                else:self.dropped_spans += 1
            self.active.clear()
        self.sample()
        return self.report()

    def report(self):
        with self.lock:
            result = dict(schema=SCHEMA, status=self.status, error_type=self.error_type,
                request=self.request, binding=self.binding, elapsed_seconds=self.ended,
                first_codes_seconds=self.marks.get("first_codes", {}).get("seconds"),
                first_pcm_seconds=self.marks.get("first_pcm", {}).get("seconds"),
                marks=self.marks, phases=self.spans, backends=self.backends,
                resources=dict(samples=self.samples, sampled_peaks=self.peaks,
                    minimum_system_available_bytes=self.minimum_available,
                    first=self.first_sample, last=self.last_sample, probe_errors=self.probe_errors,
                    scope="Sampled process/allocator peaks; provider-supplied epoch peaks retain provider scope. Missing values are unavailable, not zero."),
                truncated=dict(phases=self.dropped_spans, backends=self.dropped_backends))
            return _json(result)


def artifact_identity(path, *, sha256=None, hash_contents=False):
    """Describe an artifact; strong recommendation requires a verified full SHA.

    ``sha256`` is a caller-supplied, previously verified manifest/full-file SHA.
    Otherwise hash_contents=True performs a real, bounded-memory file read.
    Fast stat-only identities are diagnostic and cannot qualify a saved profile.
    """
    p = Path(path).resolve(strict=True)
    if not p.is_file():raise ValueError("Artifact must be a file")
    before = p.stat()
    if sha256 is not None and (not isinstance(sha256, str) or not _HEX.fullmatch(sha256)):
        raise ValueError("Artifact SHA256 must be64 lowercase hex characters")
    if hash_contents:
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024), b""):h.update(block)
        actual = h.hexdigest()
        if sha256 is not None and sha256 != actual:raise ValueError("Artifact checksum differs")
        sha256 = actual
    after = p.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Artifact changed during identity check")
    return dict(path=str(p), bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                sha256=sha256, hash_kind="full_sha256" if sha256 is not None else "stat_only")


def binding_key(binding, *, require_strong=False):
    value = _json(binding)
    required = {"machine", "gpu", "torch", "cuda", "transformers", "models", "source", "backends"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("Incomplete machine/GPU/dependency/model/source/backend binding")
    if require_strong:
        if value["transformers"] != "4.57.1":raise ValueError("Only tested Transformers4.57.1 profiles qualify")
        if not all(value[k] for k in required):raise ValueError("Empty profile binding")
        models = value["models"]
        if not isinstance(models, dict) or not {"model", "codec", "text_head", "line_head"} <= set(models):
            raise ValueError("Bind model, Codec and both control-head selections")
        if models["model"] is None or models["codec"] is None or not (models["text_head"] or models["line_head"]):
            raise ValueError("Model, Codec and an active control head are required")
        for identity in models.values():
            if identity is not None and (not isinstance(identity, dict) or not _HEX.fullmatch(str(identity.get("sha256", "")))):
                raise ValueError("Stat-only/unverified artifact cannot qualify a profile")
        if not _HEX.fullmatch(str(value["source"])):raise ValueError("Bind the source digest")
    return hashlib.sha256(_encoded(value)).hexdigest()


def _settings(value):
    v = _json(value)
    if not isinstance(v, dict) or not set(v) <= SETTINGS:raise ValueError("Benchmark may change supported backend/cache settings only")
    for name, allowed in (("gdn_backend", {"torch", "fla"}), ("moe_backend", {"auto", "torch", "cuda"}), ("expert_placement", {"ram", "vram"})):
        if name in v and v[name] not in allowed:raise ValueError("Invalid backend setting")
    if "cuda_graph" in v and type(v["cuda_graph"]) is not bool:raise ValueError("CUDA Graph must be boolean")
    if "expert_cache_gb" in v and not 0 <= _positive(v["expert_cache_gb"], "expert cache", zero=True) <= 1024:
        raise ValueError("Invalid expert cache budget")
    return v


@dataclass(frozen=True)
class BenchmarkConfig:
    id: str
    settings: Mapping[str, Any]
    stability_rank: int | None = None
    complexity: int | None = None

    def __post_init__(self):
        if not _NAME.fullmatch(self.id):raise ValueError("Invalid configuration ID")
        settings = _settings(self.settings)
        if self.stability_rank is None:
            object.__setattr__(self, "stability_rank", 3 * (settings.get("gdn_backend") == "fla")
                + 2 * bool(settings.get("cuda_graph")) + (settings.get("moe_backend") == "cuda")
                + (settings.get("expert_placement") == "vram"))
        if self.complexity is None:
            object.__setattr__(self, "complexity", sum((settings.get("gdn_backend") == "fla",
                bool(settings.get("cuda_graph")), settings.get("moe_backend") == "cuda",
                settings.get("expert_placement") == "vram")))
        if any(type(n) is not int or n < 0 for n in (self.stability_rank, self.complexity)):
            raise ValueError("Configuration ranks must be nonnegative integers")
        object.__setattr__(self, "settings", settings)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    payload: Mapping[str, Any]
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if not _NAME.fullmatch(self.id) or not all(_NAME.fullmatch(t) for t in self.tags):raise ValueError("Invalid case ID/tags")
        object.__setattr__(self, "payload", _json(self.payload))
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))


@dataclass(frozen=True)
class BenchmarkRun:
    requested_load_state: str
    repetition: int


@dataclass(frozen=True)
class BenchmarkOutcome:
    audio_seconds: float
    artifact: Mapping[str, Any]
    completed: bool = True


@dataclass(frozen=True)
class QualityVerdict:
    passed: bool
    checks: Mapping[str, bool]
    details: Mapping[str, Any] = field(default_factory=dict)


class BenchmarkCancelled(RuntimeError):
    pass


class BenchmarkUnavailable(RuntimeError):
    """Caller certifies this configuration is unavailable before unsafe work."""


def compare_performance(previous, current):
    """Compare two completed observations with the same model/environment binding."""
    if not previous.get("binding", {}).get("models") or _encoded(previous["binding"]) != _encoded(current.get("binding")):
        raise ValueError("Consecutive-job comparison requires the same model/environment binding")
    def delta(a, b):
        return None if a is None or b is None else b-a
    result = dict(same_binding=True,
        elapsed_seconds_delta=delta(previous.get("elapsed_seconds"),current.get("elapsed_seconds")),
        first_codes_seconds_delta=delta(previous.get("first_codes_seconds"),current.get("first_codes_seconds")),
        first_pcm_seconds_delta=delta(previous.get("first_pcm_seconds"),current.get("first_pcm_seconds")))
    a, b = previous.get("resources", {}).get("sampled_peaks", {}), current.get("resources", {}).get("sampled_peaks", {})
    result["sampled_peak_byte_deltas"] = {k:delta(a.get(k),b.get(k)) for k in _PEAKS}
    return result


def run_benchmark(binding, cases, configs, run_case, quality_gate=None, *, quality_batch_gate=None, prepare=None,
                  warm_repeats=2, include_cold=True, near_fastest_ratio=1.05,
                  required_coverage=REQUIRED_COVERAGE, probe=None, clock=None,
                  sample_interval=.1, cancelled=None, progress=None):
    """Run actual serial callbacks, then validate saved outputs without warming bias.

    run_case(config, case, recorder, BenchmarkRun) must complete its engine work
    and return BenchmarkOutcome with JSON artifact paths/metadata, never audio
    tensors. prepare(config, case, run) owns engine reset for a cold request;
    actual cold/warm load observations belong in recorder.phase(). Quality runs
    only after ALL timed jobs, so ASR loading cannot contaminate warm repeats.
    Alternatively quality_batch_gate receives all (case, outcome) pairs once
    and returns one QualityVerdict per pair, suitable for one shared ASR engine.
    Unexpected execution errors stop the matrix; unsafe CUDA recovery is not
    guessed. BenchmarkUnavailable skips a configuration; cancellation selects no
    profile. No input/sampling/precision or model change is allowed in settings.
    """
    key = binding_key(binding, require_strong=True)
    if (quality_gate is None) == (quality_batch_gate is None):
        raise ValueError("Supply exactly one single-output or batch quality callback")
    cases, configs = list(cases), list(configs)
    if (not cases or not configs or not all(isinstance(c,BenchmarkCase) for c in cases)
        or not all(isinstance(c,BenchmarkConfig) for c in configs)
        or len({c.id for c in cases}) != len(cases) or len({c.id for c in configs}) != len(configs)):
        raise ValueError("Unique nonempty cases/configurations are required")
    if type(warm_repeats) is not int or not 2 <= warm_repeats <= 10:raise ValueError("Two to ten consecutive warm jobs are required")
    if len(cases)>32 or len(configs)>32 or len(cases)*len(configs)*(warm_repeats+int(include_cold))>1024:
        raise ValueError("Benchmark matrix exceeds the bounded run budget")
    if type(include_cold) is not bool or not 1 <= _positive(near_fastest_ratio, "near-fastest ratio") <= 1.05:
        raise ValueError("Near-fastest ratio must be within five percent")
    coverage = set().union(*(set(c.tags) for c in cases))
    required_coverage = REQUIRED_COVERAGE | set(required_coverage)
    if not required_coverage <= coverage:raise ValueError("Benchmark lacks required short/long/reference/multi-speaker coverage")
    clock = clock or time.perf_counter
    report = dict(schema=BENCHMARK_SCHEMA, binding=_json(binding), binding_key=key,
        status="running", coverage=sorted(coverage), cases=[dict(id=c.id, payload=_json(c.payload), tags=c.tags) for c in cases],
        configs=[dict(id=c.id, settings=_json(c.settings), stability_rank=c.stability_rank, complexity=c.complexity) for c in configs],
        results=[], recommendation=None, near_fastest_ratio=near_fastest_ratio,
        warm_repeats=warm_repeats, include_cold=include_cold, required_coverage=sorted(required_coverage),
        timing_scope="Own measured callback wall time/audio duration; quality time excluded. Caller must complete actual work before returning.")
    pending, previous_jobs = [], {}
    def check_cancel():
        if cancelled is not None and cancelled():raise BenchmarkCancelled("Benchmark cancelled")
    try:
        for config in configs:
            unavailable = False
            for case in cases:
                if unavailable:break
                plan = ([BenchmarkRun("cold", 0)] if include_cold else []) + [BenchmarkRun("warm", i) for i in range(warm_repeats)]
                for run in plan:
                    check_cancel()
                    cc = BenchmarkConfig(config.id, config.settings, config.stability_rank, config.complexity)
                    sc = BenchmarkCase(case.id, case.payload, case.tags)
                    frozen = _encoded([cc.settings, sc.payload])
                    row = dict(config_id=config.id, case_id=case.id, requested_load_state=run.requested_load_state, repetition=run.repetition, status="running")
                    report["results"].append(row)
                    recorder = PerformanceRecorder(case.payload, binding, probe=probe, clock=clock, sample_interval=sample_interval)
                    if progress:progress(_json(row))
                    try:
                        if prepare:prepare(cc, sc, run)
                        check_cancel()
                        with recorder:
                            outcome = run_case(cc, sc, recorder, run)
                            check_cancel()
                            if not isinstance(outcome, BenchmarkOutcome) or outcome.completed is not True:
                                raise ValueError("Callback must return a completed BenchmarkOutcome")
                            duration = _positive(outcome.audio_seconds, "generated audio duration")
                            artifact = _json(outcome.artifact)
                            if not isinstance(artifact, dict) or not artifact:raise ValueError("Real output artifact metadata is required")
                            if _encoded([cc.settings, sc.payload]) != frozen:raise ValueError("Callback mutated fixed settings or input")
                        diagnostic = recorder.report()
                        elapsed = _positive(diagnostic["elapsed_seconds"], "measured callback time")
                        row.update(status="ok", elapsed_seconds=elapsed, audio_seconds=duration,
                                   rtf=elapsed/duration, diagnostics=diagnostic, artifact=artifact)
                        previous = previous_jobs.get((config.id,case.id))
                        if previous is not None:row["consecutive"] = compare_performance(previous,diagnostic)
                        previous_jobs[config.id,case.id] = diagnostic
                        pending.append((case, outcome, row))
                    except BenchmarkUnavailable as exc:
                        recorder.finish("unavailable", error_type=type(exc).__name__)
                        row.update(status="unavailable", error_type=type(exc).__name__)
                        unavailable = True
                        break
                    except BenchmarkCancelled:
                        recorder.finish("cancelled", error_type="BenchmarkCancelled")
                        row["status"] = "cancelled"
                        raise
                    except Exception as exc:
                        recorder.finish("error", error_type=type(exc).__name__)
                        row.update(status="error", error_type=type(exc).__name__, diagnostics=recorder.report())
                        report["status"] = "error"
                        return _json(report)
        batched = None
        quality_cases = [BenchmarkCase(case.id,case.payload,case.tags) for case,_,_ in pending]
        if quality_batch_gate is not None:
            check_cancel()
            fixed = [_encoded(c.payload) for c in quality_cases]
            start = clock()
            batched = list(quality_batch_gate([(c,outcome) for c,(_,outcome,_) in zip(quality_cases,pending)]))
            check_cancel()
            if len(batched)!=len(pending) or fixed!=[_encoded(c.payload) for c in quality_cases]:
                raise ValueError("Batch quality callback changed case inputs or result count")
            report["quality_batch_seconds"] = max(0.,clock()-start)
        for index,(case, outcome, row) in enumerate(pending):
            check_cancel()
            started = clock()
            quality_case = quality_cases[index]
            frozen_case = _encoded(quality_case.payload)
            verdict = batched[index] if batched is not None else quality_gate(quality_case, outcome)
            check_cancel()
            if _encoded(quality_case.payload) != frozen_case:raise ValueError("Quality callback mutated the fixed case")
            if not isinstance(verdict, QualityVerdict) or type(verdict.passed) is not bool:
                raise ValueError("Quality callback must return QualityVerdict")
            checks = _json(verdict.checks)
            if not isinstance(checks, dict) or not checks or any(type(v) is not bool for v in checks.values()):
                raise ValueError("Explicit nonempty boolean quality checks are required")
            row["quality"] = dict(passed=verdict.passed and all(checks.values()), checks=checks,
                details=_json(verdict.details), elapsed_seconds=None if batched is not None else max(0., clock()-started),
                timing_scope="shared_batch" if batched is not None else "single_output")
        report["status"] = "completed"
    except BenchmarkCancelled:
        report["status"] = "cancelled"
        return _json(report)
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__)
        return _json(report)
    report["recommendation"] = recommend_profile(report)
    return _json(report)


def recommend_profile(report):
    """Quality first; among configurations within5%, prefer supplied stable/simple ranks."""
    if report.get("schema") != BENCHMARK_SCHEMA or report.get("status") != "completed":return None
    if binding_key(report["binding"], require_strong=True) != report.get("binding_key"):
        raise ValueError("Benchmark binding digest changed")
    if not REQUIRED_COVERAGE <= set(report.get("coverage", [])):return None
    repeats=report.get("warm_repeats")
    if type(repeats) is not int or repeats<2:return None
    cases = {c["id"] for c in report["cases"]}
    eligible = []
    for config in report["configs"]:
        rows = [r for r in report["results"] if r["config_id"] == config["id"]]
        if not rows or any(r["status"] != "ok" or not r.get("quality", {}).get("passed")
            or not r.get("quality",{}).get("checks") or not all(v is True for v in r["quality"]["checks"].values()) for r in rows):continue
        if len(rows)!=len(cases)*(repeats+int(report.get("include_cold",False))):continue
        if any(not r["diagnostics"]["backends"] or r["diagnostics"]["truncated"]["backends"]
            or any(b.get("fallback_reason") or b.get("actual") is None for b in r["diagnostics"]["backends"]) for r in rows):continue
        if any(not _observed_settings(config["settings"],r["diagnostics"]["backends"]) for r in rows):continue
        scores, drift = [], {}
        for case in sorted(cases):
            warm = [r for r in rows if r["case_id"] == case and r["requested_load_state"] == "warm"]
            if len(warm) != repeats:break
            values = [_positive(r["rtf"], "RTF") for r in warm]
            if any(not math.isclose(r["rtf"],_positive(r["elapsed_seconds"],"elapsed time")/_positive(r["audio_seconds"],"audio duration"),rel_tol=1e-12) for r in warm):
                raise ValueError("RTF differs from measured time/audio duration")
            scores.append(statistics.median(values))
            drift[case] = dict(first_rtf=values[0], last_rtf=values[-1], relative_change=values[-1]/values[0]-1)
        else:
            eligible.append(dict(config_id=config["id"], settings=_settings(config["settings"]),
                score=statistics.mean(scores), stability_rank=config["stability_rank"], complexity=config["complexity"],
                warm_consecutive=drift, quality_passed=True))
    if not eligible:return None
    fastest = min(c["score"] for c in eligible)
    ratio = report.get("near_fastest_ratio", 1.05)
    if type(ratio) not in (int,float) or not 1 <= ratio <= 1.05:raise ValueError("Invalid fastest-band policy")
    close = [c for c in eligible if c["score"] <= fastest*ratio]
    winner = min(close, key=lambda c:(c["stability_rank"],c["complexity"],c["score"],c["config_id"]))
    return dict(winner, fastest_score=fastest, near_fastest_ratio=ratio,
                score_kind="mean of per-case median warm wall-time/audio-duration ratios")


def _observed_settings(settings, backends):
    aliases = {"moe_backend":{"moe_backend","moe"}, "gdn_backend":{"gdn_backend","gdn"},
        "cuda_graph":{"cuda_graph","graph"}, "expert_placement":{"expert_placement","placement"}}
    for name, names in aliases.items():
        if name not in settings:continue
        matches = [b for b in backends if b["component"] in names]
        if not matches:return False
        if settings[name] != "auto" and any(b["actual"]!=settings[name] for b in matches):return False
    return True


class BenchmarkProfileStore:
    """Local JSON profiles; exact binding mismatch is a cache miss, never auto-applied."""
    def __init__(self, directory):
        self.directory = Path(directory)

    def save(self, report):
        try:
            recommendation = recommend_profile(report)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid benchmark qualification") from exc
        if recommendation is None:raise ValueError("A completed quality-gated benchmark recommendation is required")
        key = binding_key(report["binding"], require_strong=True)
        profile = dict(schema=PROFILE_SCHEMA, binding=_json(report["binding"]), binding_key=key,
            recommendation=recommendation, qualification=dict(schema=BENCHMARK_SCHEMA,
                report_sha256=hashlib.sha256(_encoded(report)).hexdigest(), quality_passed=True,
                cases=[c["id"] for c in report["cases"]], coverage=_json(report["coverage"])))
        profile["integrity_sha256"] = hashlib.sha256(_encoded(profile)).hexdigest()
        data = _encoded(profile)
        if len(data) > 1024*1024:raise ValueError("Profile is too large")
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".profile-", suffix=".tmp", delete=False) as f:
                temporary = Path(f.name)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.directory/(key+".json"))
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
        return _json(profile)

    def load(self, binding):
        try:
            key = binding_key(binding, require_strong=True)
            p = self.directory/(key+".json")
            if p.stat().st_size > 1024*1024:return None
            value = json.loads(p.read_text("utf-8"))
            integrity = value.get("integrity_sha256")
            if integrity != hashlib.sha256(_encoded({k:v for k,v in value.items() if k!="integrity_sha256"})).hexdigest():return None
            if value.get("schema") != PROFILE_SCHEMA or value.get("binding_key") != key:return None
            if binding_key(value["binding"], require_strong=True) != key:return None
            if value["qualification"].get("quality_passed") is not True:return None
            if not REQUIRED_COVERAGE <= set(value["qualification"].get("coverage",[])):return None
            r = value["recommendation"]
            if r.get("quality_passed") is not True or not _HEX.fullmatch(value["qualification"].get("report_sha256", "")):return None
            _positive(r["score"], "profile score")
            _settings(r["settings"])
            return _json(value)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None


__all__ = ["PerformanceRecorder", "ResourceProbe", "artifact_identity", "binding_key",
    "BenchmarkCase", "BenchmarkConfig", "BenchmarkRun", "BenchmarkOutcome", "QualityVerdict",
    "BenchmarkCancelled", "BenchmarkUnavailable", "run_benchmark", "recommend_profile", "compare_performance", "BenchmarkProfileStore"]
