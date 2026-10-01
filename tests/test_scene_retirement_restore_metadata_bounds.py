"""ADP-009D/day28: reject extension allocation before any restore destination."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_restore.py
import tarfile

import pytest

from tests.test_scene_retirement_member_mutation import setup_operation


@pytest.mark.parametrize('kind', [tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME,
                                 tarfile.GNUTYPE_LONGLINK, tarfile.GNUTYPE_SPARSE])
def test_unbounded_tar_extension_refuses_before_parser_allocation(tmp_path, monkeypatch, kind):
    from blueprint_pipeline import task_evaluation_scene_retirement_restore as restore
    _, member, preserved, journal = setup_operation(tmp_path, monkeypatch)
    header = tarfile.TarInfo('owned-extension-probe')
    header.type, header.size = kind, 16 * 1024 * 1024
    raw = header.tobuf(format=tarfile.GNU_FORMAT)
    requested = []
    actual_read = tarfile._Stream.read
    def stream_read(stream, size):
        if size > 8192:
            requested.append(size)
            raise RuntimeError('parser attempted unbounded extension allocation')
        return actual_read(stream, size)
    monkeypatch.setattr(tarfile._Stream, 'read', stream_read)
    class Transport:
        def read_archive(self, uri):
            # Tiny real header fragments; no large archive or payload allocation.
            for offset in range(0, len(raw), 64):
                yield raw[offset:offset + 64]
    selected = dict(preserved, archive=dict(preserved['archive'], size_bytes=len(raw),
                                            sha256='sha256:' + 'a' * 64))
    with pytest.raises(ValueError, match='scene_retirement_readback_unproven'):
        restore._consume(selected, Transport(), journal.allowance)
    assert requested == []
    assert (member / 'nested/evidence.bin').read_bytes() == b'preserved-evidence'


def test_generated_pax_long_path_roundtrip_preserves_actual_payload(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    from tests.test_scene_retirement_preservation import MemoryTransport
    access, member, _, journal = setup_operation(tmp_path, monkeypatch)
    long_path = member / ('long-path-' + 'a' * 110)
    long_path.mkdir()
    payload = long_path / ('owned-' + 'b' * 120 + '.bin')
    payload.write_bytes(b'actual-generated-PAX-payload')
    transport = MemoryTransport([member])
    preserved = preserve_members([member], transport=transport, allowance=journal.allowance, token='3' * 32)
    with access.exclusive_scene_access():
        detach_and_remove(preserved, member_index=0, generation_id='2' * 32, journal=journal)
        transport.members = []
        result = restore_preserved_members(preserved, transport=transport, journal=journal)
    assert result[0]['outcome'] == 'restored'
    assert payload.read_bytes() == b'actual-generated-PAX-payload'


def test_nested_pax_refuses_before_recursive_metadata_parse(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_restore as restore
    _, member, preserved, journal = setup_operation(tmp_path, monkeypatch)
    extension = tarfile.TarInfo._create_pax_generic_header(
        {'path': '0/nested/evidence.bin'}, tarfile.XHDTYPE, 'utf-8')
    terminal = tarfile.TarInfo('0/nested/evidence.bin').tobuf()
    raw = extension * 3 + terminal + b'\0' * 1024
    parsed = []
    original_pax = tarfile.TarInfo._proc_pax
    def pax(item, archive):
        parsed.append(item.size)
        return original_pax(item, archive)
    monkeypatch.setattr(tarfile.TarInfo, '_proc_pax', pax)
    class Transport:
        def read_archive(self, uri):
            for offset in range(0, len(raw), 64):
                yield raw[offset:offset + 64]
    selected = dict(preserved, archive=dict(preserved['archive'], size_bytes=len(raw),
                                            sha256='sha256:' + 'a' * 64))
    with pytest.raises(ValueError, match='scene_retirement_readback_unproven'):
        restore._consume(selected, Transport(), journal.allowance)
    assert len(parsed) == 1
    assert (member / 'nested/evidence.bin').read_bytes() == b'preserved-evidence'
