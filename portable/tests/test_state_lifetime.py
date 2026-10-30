"""Mapped Session storage is released without global cyclic-GC workarounds."""
import gc
import json
import struct

import pytest
import torch

from drama3_t8.state import Reference,Session,Take,load_session,save_session


def _frames():
    value=torch.zeros((2,11),dtype=torch.int64);value[:,0]=248099;value[:,1:]=7
    return value


@pytest.mark.parametrize("kind",["reference","take","typed_sources"])
def test_discarded_cpu_session_releases_mapped_file_with_cyclic_gc_disabled(tmp_path,kind):
    if kind=="reference":original=Session((Reference("source",torch.ones((10,4),dtype=torch.int64)),))
    else:
        source=Take("source","source",_frames())
        take=source if kind=="take" else Take("result","result",_frames(),sources=(source,),blobs={"preview":torch.ones((1,64))})
        original=Session(takes=(take,))
    path=tmp_path/"session.safetensors";save_session(original,path);data=path.read_bytes()
    enabled=gc.isenabled();gc.disable()
    try:
        restored=load_session(path)
        assert restored.session_id==original.session_id
        if kind=="typed_sources":
            assert torch.equal(restored.takes[0].sources[0].frames,original.takes[0].sources[0].frames)
            assert torch.equal(restored.takes[0].blobs["preview"],original.takes[0].blobs["preview"])
        del restored
        # Windows rejects this with OSError22 while a stale mmap survives.
        # Linux also executes a real read/write lifecycle, without any gc.collect.
        assert path.write_bytes(data)==len(data)
        assert path.read_bytes()==data
    finally:
        if enabled:gc.enable()


def test_repeated_reference_loads_do_not_require_cyclic_gc_for_release(tmp_path):
    path=tmp_path/"reference.safetensors"
    save_session(Session((Reference("source",torch.ones((10,4),dtype=torch.int64)),)),path)
    data=path.read_bytes();enabled=gc.isenabled();gc.disable()
    try:
        for _ in range(20):
            load_session(path)
            assert path.write_bytes(data)==len(data)
    finally:
        if enabled:gc.enable()


def test_rejected_typed_source_archive_releases_mapping_after_error_handling_without_gc(tmp_path):
    path=tmp_path/"invalid-source.safetensors";save_session(Session(takes=(Take("source","source",_frames()),)),path)
    data=path.read_bytes();size=struct.unpack("<Q",data[:8])[0];header=json.loads(data[8:8+size])
    metadata=json.loads(header["__metadata__"]["drama3_t8_session"])
    metadata.update(version=2,features=["typed_takes_v1"]);metadata["takes"][0]["sources"]=["f"*32]
    header["__metadata__"]["drama3_t8_session"]=json.dumps(metadata)
    encoded=json.dumps(header,separators=(",",":")).encode();encoded+=b" "*(-len(encoded)%8)
    bad=struct.pack("<Q",len(encoded))+encoded+data[8+size:];path.write_bytes(bad)
    def reject():
        try:load_session(path)
        except ValueError:return
        raise AssertionError("Corrupt graph must be rejected")
    enabled=gc.isenabled();gc.disable()
    try:
        reject()
        assert path.write_bytes(bad)==len(bad)
    finally:
        if enabled:gc.enable()
