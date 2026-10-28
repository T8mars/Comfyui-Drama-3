"""Ownership helpers must resolve the real ComfyUI custom-node package."""
import builtins
import importlib
import importlib.machinery
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "comfyui-drama3-T8"
PACKAGE = "custom_nodes.ComfyUI-Drama3-T8.drama3_t8"


@pytest.fixture
def nested_native():
    """Load actual mirror files, without running the node registration entrypoint.

    A top-level package may already be cached by the CPU bootstrap. Blocking
    its absolute imports still exposes the production namespace failure.
    """
    previous = {name: module for name, module in sys.modules.items()
                if name == PACKAGE or name.startswith(PACKAGE + ".")}
    parents = [
        ("custom_nodes", NODE.parent),
        ("custom_nodes.ComfyUI-Drama3-T8", NODE),
        (PACKAGE, NODE / "drama3_t8"),
    ]
    original_import = builtins.__import__
    patches = pytest.MonkeyPatch()

    def nested_only(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == "drama3_t8" or name.startswith("drama3_t8.")):
            raise ModuleNotFoundError("Top-level drama3_t8 import forbidden", name=name)
        return original_import(name, globals, locals, fromlist, level)

    try:
        for name in list(previous):
            sys.modules.pop(name, None)
        for name, path in parents:
            package = ModuleType(name)
            package.__package__ = name
            package.__path__ = [str(path)]
            package.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
            patches.setitem(sys.modules, name, package)
            if "." in name:
                parent, child = name.rsplit(".", 1)
                patches.setattr(sys.modules[parent], child, package, raising=False)
        patches.setattr(builtins, "__import__", nested_only)
        own = importlib.import_module(PACKAGE + ".native.checkpoint_ownership")
        loader = importlib.import_module(PACKAGE + ".native.loader")
        offload = importlib.import_module(PACKAGE + ".native.offload")
        assert Path(own.__file__).resolve() == (NODE / "drama3_t8/native/checkpoint_ownership.py").resolve()
        assert loader.load_checkpoint_storage.__module__ == loader.__name__
        assert offload.QuantHostBank.__module__ == offload.__name__
        yield SimpleNamespace(own=own, loader=loader, offload=offload)
    finally:
        patches.undo()
        for name in list(sys.modules):
            if name == PACKAGE or name.startswith(PACKAGE + "."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def _empty_model():
    # No checkpoint allocation is needed to exercise the actual helper imports.
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    return model


def test_nested_clone_contract_uses_actual_loader_and_retains_identity_checks(nested_native, monkeypatch):
    own, loader = nested_native.own, nested_native.loader
    origins = own._clone_contract()
    assert Path(origins["assign_function"]["path"]).resolve() == Path(loader.__file__).resolve()
    assert origins["assign_function"]["ast_sha256"] == own._ASSIGN_AST_SHA
    # Keep the real function code, AST and filename; only corrupt its identity.
    monkeypatch.setattr(loader.load_checkpoint_storage, "__module__", "drama3_t8.native.loader")
    with pytest.raises(own.OwnershipUnavailable, match="actual assign function contract"):
        own._clone_contract()


def test_nested_inventory_resolves_actual_offload_helpers(nested_native):
    assert nested_native.own._inventory(_empty_model(), {}, {}) == ()


def test_nested_derived_params_resolves_actual_offload_and_loader(nested_native):
    own, loader = nested_native.own, nested_native.loader
    manifest, derived, origins = own._derive_recognized_params(_empty_model(), {}, {}, {})
    assert manifest == {} and derived == []
    assert Path(origins["assign_function"]["path"]).resolve() == Path(loader.__file__).resolve()
