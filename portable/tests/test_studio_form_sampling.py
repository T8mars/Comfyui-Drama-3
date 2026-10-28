"""Native HTML number constraints must accept Studio defaults and presets.

This static contract needs no Studio, Torch, browser, or JavaScript runtime.
Preset values and their target input order are read from the shipped script.
"""
import ast
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
import re


WEB = Path(__file__).resolve().parents[1] / "web"


class _GenerateForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.presets = set()
        self.in_form = False
        self.in_presets = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.in_form = attrs.get("id") == "generate-form"
        if not self.in_form:
            return
        if tag == "input" and attrs.get("type") == "number":
            self.inputs[attrs["id"]] = attrs
        elif tag == "select":
            self.in_presets = attrs.get("id") == "sampling-preset"
        elif tag == "option" and self.in_presets:
            self.presets.add(attrs["value"])

    def handle_endtag(self, tag):
        if tag == "form":
            self.in_form = False
        elif tag == "select":
            self.in_presets = False


def _form():
    parsed = _GenerateForm()
    parsed.feed((WEB / "index.html").read_text(encoding="utf-8"))
    assert {"temperature", "top-p", "top-k", "max-frames"} <= parsed.inputs.keys()
    return parsed


def _number(value):
    number = Decimal(value)
    assert number.is_finite(), f"Non-finite HTML numeric value: {value!r}"
    return number


def _constraint_errors(name, attrs, value):
    """HTML number step base is min, then the value attribute, then zero."""
    value = _number(value)
    minimum = _number(attrs["min"]) if "min" in attrs else None
    maximum = _number(attrs["max"]) if "max" in attrs else None
    errors = []
    if minimum is not None and value < minimum:
        errors.append(f"{name}={value}: rangeUnderflow (min={minimum})")
    if maximum is not None and value > maximum:
        errors.append(f"{name}={value}: rangeOverflow (max={maximum})")
    if attrs.get("step") != "any":
        step = _number(attrs.get("step", "1"))
        assert step > 0, f"{name}: invalid step {step}"
        base = minimum if minimum is not None else _number(attrs.get("value", "0"))
        if (value - base) % step:
            errors.append(f"{name}={value}: stepMismatch (base={base}, step={step})")
    return errors


def test_studio_numeric_defaults_can_submit_without_native_constraint_failure():
    form = _form()
    # Preserve the existing sampling bounds/defaults while fixing their grid.
    for name, maximum in (("temperature", "2"), ("top-p", "1")):
        attrs = form.inputs[name]
        assert tuple(_number(attrs[key]) for key in ("min", "max", "value")) == (
            Decimal("0.01"), Decimal(maximum), Decimal("0.7"))
    errors = [error for name, attrs in form.inputs.items() if "value" in attrs
              for error in _constraint_errors(name, attrs, attrs["value"])]
    assert not errors, "Default form blocks submission: " + "; ".join(errors)


def _assert_preset_can_submit(name, expected):
    form = _form()
    assert name in form.presets
    source = (WEB / "sampling-controls.js").read_text(encoding="utf-8")
    targets = re.search(r"const\s+samplingInputs\s*=\s*(\[[^\]]+\])\.map", source)
    values = re.search(rf"\b{re.escape(name)}\s*:\s*\[([^\]]+)\]", source)
    assert targets and values, "Sampling preset binding is unavailable"
    fields = ast.literal_eval(targets.group(1))
    actual = tuple(_number(part.strip()) for part in values.group(1).split(","))
    assert fields == ["temperature", "top-p", "top-k"]
    assert actual == tuple(Decimal(value) for value in expected)
    errors = [error for field, value in zip(fields, actual)
              for error in _constraint_errors(field, form.inputs[field], value)]
    assert not errors, f"{name} preset blocks submission: " + "; ".join(errors)


def test_studio_stable_preset_can_submit_without_native_constraint_failure():
    _assert_preset_can_submit("stable", ("0.7", "0.7", "30"))


def test_studio_expressive_preset_can_submit_without_native_constraint_failure():
    _assert_preset_can_submit("expressive", ("1.0", "0.9", "0"))
