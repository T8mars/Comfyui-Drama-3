"""Only DRAMA standalone rewrites the optional Kitchen CUDA warning."""
import ast
from pathlib import Path
from types import ModuleType,SimpleNamespace
import logging
import sys
import pytest

@pytest.mark.parametrize('standalone',[True,False])
def test_optional_quant_backend_keeps_safe_disable_and_host_warning(monkeypatch,caplog,standalone):
    source=Path(__file__).resolve().parents[1]/'vendor/ComfyUI/comfy/quant_ops.py'
    block=next(n for n in ast.parse(source.read_text('utf-8')).body if isinstance(n,ast.Try))
    disabled=[];kitchen=ModuleType('comfy_kitchen');tensor=ModuleType('comfy_kitchen.tensor')
    kitchen.registry=SimpleNamespace(disable=disabled.append);kitchen.list_backends=lambda:{}
    for node in ast.walk(block):
        if isinstance(node,ast.ImportFrom) and node.module=='comfy_kitchen.tensor':
            for name in node.names:setattr(tensor,name.name,type(name.name,(),{}))
    monkeypatch.setitem(sys.modules,'comfy_kitchen',kitchen)
    monkeypatch.setitem(sys.modules,'comfy_kitchen.tensor',tensor)
    namespace={'torch':SimpleNamespace(version=SimpleNamespace(cuda='12.8')),
        'args':SimpleNamespace(drama_standalone_engine=standalone,enable_triton_backend=False,disable_triton_backend=False),
        'logging':logging}
    with caplog.at_level(logging.INFO):exec(compile(ast.Module(body=[block],type_ignores=[]),str(source),'exec'),namespace)
    assert namespace['_CK_AVAILABLE'] and disabled==['cuda','triton']
    if standalone:
        assert 'DRAMA standalone' in caplog.text and 'it is required that you update' not in caplog.text
    else:assert 'it is required that you update' in caplog.text
