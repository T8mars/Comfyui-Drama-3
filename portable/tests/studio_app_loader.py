"""Load our HTTP entry point independently of ComfyUI's unrelated app package."""
import importlib.util
from pathlib import Path
import sys

name="drama3_studio_http_tests"
if name in sys.modules:
    application=sys.modules[name]
else:
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'app.py')
    application=importlib.util.module_from_spec(spec)
    sys.modules[name]=application
    spec.loader.exec_module(application)
