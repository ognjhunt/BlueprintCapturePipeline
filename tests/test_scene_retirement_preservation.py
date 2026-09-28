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
    first.mkdir()
    second.mkdir()
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


def test_transport_cannot_claim_success_after_abandoning_partial_member_union(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members, ActionAllowance
    members = fixture_members(tmp_path)
    class PartialTransport(MemoryTransport):
        def put_archive(self,key,chunks):
            return super().put_archive(key,[next(iter(chunks))])
    with pytest.raises(ValueError,match='scene_retirement_archive_incomplete'):
        preserve_members(members,transport=PartialTransport(members),
            allowance=ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0),token='1'*32)
    assert all(path.exists() for path in members)


@pytest.mark.parametrize('first',[float('nan'),float('inf'),True,None])
@pytest.mark.parametrize('kind',['monotonic','wall'])
def test_invalid_initial_action_clock_refuses_before_establishing_origin(first,kind):
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    values = iter([first,0.0,9999.0])
    samples = []
    def selected():
        value = next(values)
        samples.append(value)
        return value
    kwargs = {'monotonic':selected,'now':lambda:200} if kind == 'monotonic' else {'now':selected,'monotonic':lambda:0}
    with pytest.raises(ValueError,match='scene_retirement_clock_unproven'):
        ActionAllowance(expires_at=1000,elapsed_seconds=1,**kwargs)
    assert len(samples) == 1


def test_raising_initial_action_clock_is_fixed_refusal_and_cannot_reset_object():
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    def fault():
        raise OSError('raw private callback detail must not escape')
    value = ActionAllowance.__new__(ActionAllowance)
    with pytest.raises(ValueError,match='scene_retirement_clock_unproven'):
        value.__init__(expires_at=1000,now=lambda:200,monotonic=fault)
    with pytest.raises(ValueError,match='scene_retirement_allowance_already_initialized'):
        value.__init__(expires_at=1000,now=lambda:200,monotonic=lambda:0)


class ChargedMemoryTransport(MemoryTransport):
    """Test the explicit installed-reader contract, not a safety flag."""
    def __init__(self,members,allowance):
        super().__init__(members)
        self.allowance=allowance
        self.origins=[]
    def read_archive(self,uri):
        pytest.fail('native charged reader was bypassed')
    def read_archive_charged(self,uri,allowance):
        assert allowance is self.allowance
        self.origins.append(allowance)
        for chunk in MemoryTransport.read_archive(self,uri):
            allowance.charge('remote_bytes',len(chunk))
            yield chunk


def test_preservation_uses_native_precharged_reader_with_one_origin_and_no_double_charge(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members,ActionAllowance
    members=fixture_members(tmp_path)
    allowance=ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0)
    transport=ChargedMemoryTransport(members,allowance)
    result=preserve_members(members,transport=transport,allowance=allowance,token='1'*32)
    assert transport.origins==[allowance]
    assert allowance.counts['remote_bytes']==2*result['archive']['size_bytes']


def test_restore_archive_reader_uses_native_precharged_bytes_once_with_same_origin():
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import _ArchiveReader
    allowance=ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0)
    transport=ChargedMemoryTransport([],allowance)
    raw=b'actual bounded remote bytes'*17
    uri='s3://private-fixture/retained.tar'
    transport.objects[uri]=raw
    reader=_ArchiveReader(transport,{'uri':uri,'sha256':'sha256:'+hashlib.sha256(raw).hexdigest(),
                                     'size_bytes':len(raw)},allowance)
    assert reader.read(len(raw))==raw
    reader.verify()
    assert transport.origins==[allowance] and allowance.counts['remote_bytes']==len(raw)
