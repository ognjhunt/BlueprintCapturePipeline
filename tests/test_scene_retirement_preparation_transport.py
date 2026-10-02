"""Actual native archive transport retains finite original-token attempt keys."""
import pytest

from tests.test_scene_retirement_operator_bridge import Client, allowance, bridge


@pytest.mark.parametrize('suffix',['','.1','.256'])
def test_native_preparation_attempt_archive_fresh_readback_keeps_same_allowance(suffix):
    module=bridge()
    client=Client()
    transport=module.SceneArchiveTransport(client=client,bucket='private-artifacts')
    origin=allowance()
    transport.bind_allowance(origin)
    name='1'*32+suffix+'.tar'
    reference=transport.put_archive(name,iter([b'actual-private-bytes']))
    assert reference['uri'].endswith('/'+name)
    assert b''.join(transport.read_archive_charged(reference['uri'],origin))==b'actual-private-bytes'
    assert origin.counts['remote_bytes']==len(b'actual-private-bytes')+1
    assert client.body.closed
    transport.close()
    assert client.calls==['create','part','complete','get','close']


@pytest.mark.parametrize('suffix',['.0','.01','.257','.1000','../1','.1/foreign'])
def test_unknown_attempt_name_is_refused_before_native_upload_or_get(suffix):
    module=bridge()
    client=Client()
    transport=module.SceneArchiveTransport(client=client,bucket='private-artifacts')
    origin=allowance()
    transport.bind_allowance(origin)
    name='1'*32+suffix+'.tar'
    with pytest.raises(ValueError):
        transport.put_archive(name,iter([b'not-uploaded']))
    with pytest.raises(ValueError):
        list(transport.read_archive_charged('s3://private-artifacts/'+module._PREFIX+name,origin))
    assert client.calls==[] and origin.counts['remote_bytes']==0
    transport.close()
