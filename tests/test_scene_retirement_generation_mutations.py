"""Numeric descriptor substitution must refuse before actual producer mutations."""
import os
from contextlib import nullcontext

import pytest

from blueprint_pipeline import task_evaluation_scene_retirement_access as access
from blueprint_pipeline import task_evaluation_scene_retirement_generations as generations


@pytest.mark.parametrize('substitution', ['owned-temp-before-write', 'parent-before-create', 'temp-before-fsync'])
def test_generation_publisher_never_mutates_reused_foreign_descriptor(tmp_path, monkeypatch, substitution):
    store = tmp_path / 'store'
    store.mkdir()
    foreign = tmp_path / 'foreign'
    if substitution == 'parent-before-create':
        foreign.mkdir()
    else:
        foreign.write_bytes(b'foreign-original-bytes')
    real_open, real_write, real_fsync = os.open, os.write, os.fsync
    foreign_fd = real_open(foreign, os.O_RDONLY | os.O_DIRECTORY if foreign.is_dir() else os.O_RDWR)
    substituted = []
    try:
        cleanup_refusal = (pytest.raises(access.SceneRetirementAccessError, match='descriptor_cleanup_failed')
                           if substitution == 'parent-before-create' else nullcontext())
        with cleanup_refusal, access._opened(store, directory=True) as (parent, info):
            original_new = generations._new_file
            def new_file(fd, name, **kwargs):
                if substitution == 'parent-before-create':
                    os.close(fd)
                    os.dup2(foreign_fd, fd)
                    substituted.append(fd)
                result = original_new(fd, name, **kwargs)
                if substitution == 'owned-temp-before-write':
                    os.close(result[0])
                    os.dup2(foreign_fd, result[0])
                    substituted.append(result[0])
                return result
            def write(fd, value):
                result = real_write(fd, value)
                if substitution == 'temp-before-fsync' and not substituted:
                    os.close(fd)
                    os.dup2(foreign_fd, fd)
                    substituted.append(fd)
                return result
            def fsync(fd):
                assert access._identity(os.fstat(fd)) != access._identity(os.fstat(foreign_fd)), 'fsync used foreign descriptor'
                return real_fsync(fd)
            monkeypatch.setattr(os, 'fsync', fsync)
            monkeypatch.setattr(generations, '_new_file', new_file)
            monkeypatch.setattr(os, 'write', write)
            # New keyword is supplied only once production supports it; current
            # RED still reaches the actual unsafe operation, not a TypeError.
            import inspect
            kwargs = {'parent_identity': access._identity(info)} if 'parent_identity' in inspect.signature(generations._write).parameters else {}
            with pytest.raises(access.SceneRetirementAccessError):
                generations._write(parent, 'state.json', {'state': 'active'}, **kwargs)
            assert substituted
            for fd in substituted:
                assert access._identity(os.fstat(fd)) == access._identity(os.fstat(foreign_fd))
            if foreign.is_dir():
                assert list(foreign.iterdir()) == []
            else:
                assert foreign.read_bytes() == b'foreign-original-bytes'
    finally:
        for fd in set(substituted):
            try:
                os.close(fd)
            except OSError:
                pass
        os.close(foreign_fd)


def test_generation_metadata_publication_preserves_installed_store_owner_before_link(tmp_path,monkeypatch):
    store=tmp_path/'service-generations'
    store.mkdir(mode=0o700)
    calls=[]
    real_chown,real_link=os.fchown,os.link
    def chown(fd,uid,gid):
        calls.append((uid,gid))
        return real_chown(fd,uid,gid)
    def link(source,destination,**kwargs):
        assert calls==[(store.stat().st_uid,store.stat().st_gid)]
        return real_link(source,destination,**kwargs)
    monkeypatch.setattr(os,'fchown',chown)
    monkeypatch.setattr(os,'link',link)
    with access._opened(store,directory=True) as (parent,info):
        generations._write(parent,'state.json',{'state':'retired'},parent_identity=access._identity(info))
    info=(store/'state.json').stat()
    assert info.st_uid==store.stat().st_uid and info.st_gid==store.stat().st_gid
    assert info.st_mode & 0o777==0o600
