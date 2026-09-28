"""Real payload preservation must complete fresh full readback before removal."""
import hashlib

import pytest


class MemoryTransport:
    def __init__(self, members, *, corrupt=False):
        self.members = members
        self.corrupt = corrupt
        self.objects = {}
        self.readbacks = 0
    def put_archive(self, key, chunks):
        raw = b''.join(chunks)
        self.objects['s3://private-fixture/' + key] = raw
        return {'uri':'s3://private-fixture/' + key,
                'sha256':'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes':len(raw)}
    def read_archive(self, uri):
        assert all(path.exists() for path in self.members), 'removal preceded fresh complete readback'
        self.readbacks += 1
        raw = self.objects[uri]
        if self.corrupt:
            raw = b'X' + raw[1:]
        for offset in range(0, len(raw), 37):
            yield raw[offset:offset+37]


def fixture_members(tmp_path):
    first, second = tmp_path/'source', tmp_path/'sam'
    first.mkdir(); second.mkdir()
    (first/'input.bin').write_bytes(b'original-capture')
    (second/'evidence.bin').write_bytes(b'original-sam-evidence')
    return [first, second]


def test_real_members_are_streamed_and_freshly_read_back_without_mutation(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members, ActionAllowance
    members = fixture_members(tmp_path)
    transport = MemoryTransport(members)
    result = preserve_members(members, transport=transport,
        allowance=ActionAllowance(expires_at=200, now=lambda:100, monotonic=lambda:0), token='1'*32)
    assert transport.readbacks == 1
    assert result['archive']['sha256'] == result['archive']['fresh_readback_sha256']
    assert {(row['member_index'], row['relative_path']) for row in result['files']} == {(0,'input.bin'),(1,'evidence.bin')}
    assert all(path.exists() for path in members)


def test_corrupt_fresh_readback_keeps_every_original_byte(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members, ActionAllowance
    members = fixture_members(tmp_path)
    with pytest.raises(ValueError, match='scene_retirement_readback_unproven'):
        preserve_members(members, transport=MemoryTransport(members,corrupt=True),
            allowance=ActionAllowance(expires_at=200, now=lambda:100, monotonic=lambda:0), token='1'*32)
    assert (members[0]/'input.bin').read_bytes() == b'original-capture'
    assert (members[1]/'evidence.bin').read_bytes() == b'original-sam-evidence'


def test_deadline_and_external_hardlink_refuse_before_transport(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members, ActionAllowance
    import os
    members = fixture_members(tmp_path)
    os.link(members[0]/'input.bin', tmp_path/'outside.bin')
    transport = MemoryTransport(members)
    with pytest.raises(ValueError, match='scene_retirement_shared_inode'):
        preserve_members(members, transport=transport,
            allowance=ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0),token='1'*32)
    assert transport.objects == {}
    with pytest.raises(ValueError, match='scene_retirement_consent_expired'):
        preserve_members(members, transport=transport,
            allowance=ActionAllowance(expires_at=99,now=lambda:100,monotonic=lambda:0),token='1'*32)
    assert transport.objects == {}
