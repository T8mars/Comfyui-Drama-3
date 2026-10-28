"""Run production cleanup bodies against CPU-only boundaries; import no Torch.

These contracts establish ordering and failure semantics, not physical RAM release.
By default they read canonical sources. Private validation can select a candidate.
"""
import ast
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
RELEASE_SOURCE = REPOSITORY / 'drama3_t8/native/acceleration.py'
ENGINE_SOURCE = REPOSITORY / 'drama3_t8/engine.py'
FIELDS = ('drama', 'codec', 'asr', 'text_head', 'text_head_patcher',
          'line_head', 'line_head_patcher')


def source_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef)
                    and node.name == class_name).body
    function = next(node for node in body if isinstance(node, ast.FunctionDef)
                    and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def helper_fixture(initialized=True, binding='legacy', sync_error=None, release_error=None):
    calls = []

    def synchronize():
        calls.append('sync')
        if sync_error is not None:
            raise sync_error

    def release():
        calls.append('release')
        if release_error is not None:
            raise release_error

    fake = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: initialized, synchronize=synchronize),
        accelerator=SimpleNamespace(), _C=SimpleNamespace())
    if binding == 'public':
        fake.accelerator.empty_host_cache = release
        # A public binding must win over both legacy fallbacks.
        fake._C._host_emptyCache = lambda: calls.append('wrong_legacy')
        fake._C._accelerator_emptyHostCache = lambda: calls.append('wrong_oldest')
    elif binding == 'legacy':
        fake._C._host_emptyCache = release
        fake._C._accelerator_emptyHostCache = lambda: calls.append('wrong_oldest')
    elif binding == 'oldest':
        fake._C._accelerator_emptyHostCache = release
    return source_function(RELEASE_SOURCE, 'release_unused_pinned_memory', {'torch': fake}), calls


class PinnedHostReleaseOrderTests(unittest.TestCase):
    def test_uninitialized_does_not_synchronize_or_release(self):
        release, calls = helper_fixture(initialized=False)
        self.assertIs(release(), False)
        self.assertEqual(calls, [])

    def test_missing_binding_does_not_synchronize(self):
        release, calls = helper_fixture(binding=None)
        self.assertIs(release(), False)
        self.assertEqual(calls, [])

    def test_each_supported_binding_synchronizes_before_release(self):
        for binding in ('public', 'legacy', 'oldest'):
            with self.subTest(binding=binding):
                release, calls = helper_fixture(binding=binding)
                self.assertIs(release(), True)
                self.assertEqual(calls, ['sync', 'release'])

    def test_sync_fault_preserves_identity_and_does_not_release_or_report_success(self):
        error = RuntimeError('synthetic CUDA synchronization failure')
        release, calls = helper_fixture(sync_error=error)
        with self.assertRaises(RuntimeError) as raised:
            release()
        self.assertIs(raised.exception, error)
        self.assertEqual(calls, ['sync'])

    def test_release_fault_preserves_identity_after_sync(self):
        error = OSError('synthetic host purge failure')
        release, calls = helper_fixture(release_error=error)
        with self.assertRaises(OSError) as raised:
            release()
        self.assertIs(raised.exception, error)
        self.assertEqual(calls, ['sync', 'release'])

    def test_real_unload_body_continues_device_cleanup_after_sync_fault(self):
        for prior_host_failure in (False, True):
            with self.subTest(prior_host_failure=prior_host_failure):
                sync_error = RuntimeError('synthetic synchronization fault')
                host_error = MemoryError('synthetic earlier host-unload fault')
                release, sync_calls = helper_fixture(sync_error=sync_error)
                calls = []
                engine = SimpleNamespace(**{name: object() for name in FIELDS})
                engine.drama = SimpleNamespace(experts=SimpleNamespace(
                    cache=None, active=False, close=lambda: calls.append('experts')))

                def host_unload():
                    calls.append('host')
                    if prior_host_failure:
                        raise host_error

                def collect():
                    self.assertTrue(all(getattr(engine, name) is None for name in FIELDS))
                    calls.append('gc')

                def purge():
                    calls.append('pins')
                    return release()

                package_name = '_drama_host_release_contract'
                package = ModuleType(package_name)
                package.__path__ = []
                native = ModuleType(package_name + '.native')
                native.__path__ = []
                acceleration = ModuleType(package_name + '.native.acceleration')
                acceleration.release_unused_pinned_memory = purge
                modules = {package_name: package, native.__name__: native,
                           acceleration.__name__: acceleration}
                namespace = dict(__package__=package_name,
                    mm=SimpleNamespace(unload_all_models=host_unload,
                        soft_empty_cache=lambda: calls.append('device_cache')),
                    gc=SimpleNamespace(collect=collect))
                unload = source_function(ENGINE_SOURCE, 'unload', namespace, 'DramaEngine')
                with ExitStack() as stack:
                    stack.enter_context(patch.dict(sys.modules, modules))
                    stack.enter_context(patch('logging.warning'))
                    with self.assertRaises(BaseException) as raised:
                        unload(engine)
                self.assertIs(raised.exception, host_error if prior_host_failure else sync_error)
                self.assertEqual(calls, ['experts', 'host', 'gc', 'pins', 'device_cache'])
                self.assertEqual(sync_calls, ['sync'])
                self.assertTrue(all(getattr(engine, name) is None for name in FIELDS))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=RELEASE_SOURCE)
    parser.add_argument('--engine-source', type=Path, default=ENGINE_SOURCE)
    args, remaining = parser.parse_known_args()
    RELEASE_SOURCE, ENGINE_SOURCE = args.source, args.engine_source
    assert 'torch' not in sys.modules
    unittest.main(argv=[sys.argv[0], *remaining])
