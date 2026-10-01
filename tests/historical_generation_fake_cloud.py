"""Tiny in-process object transport; never contacts a provider or credentials."""
import base64
import io
import json
import zlib

MAX_OBJECT_BYTES = 65536
MAX_WIRE_BYTES = 16384


def wire_state(cloud):
    """Compress diagnostic transport only; original archive bytes stay exact."""
    assert len(cloud.objects) <= 1
    assert all(type(raw) is bytes and len(raw) <= MAX_OBJECT_BYTES for raw in cloud.objects.values())
    value = dict(corrupt=cloud.corrupt,
        objects_zlib={key: base64.b64encode(zlib.compress(raw)).decode()
                      for key, raw in cloud.objects.items()},
        metadata=cloud.metadata, calls=cloud.calls)
    assert len(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()) + 1 <= MAX_WIRE_BYTES
    return value


def decode_wire_state(value):
    """Bound allocation before decoding the actual fixture object's bytes."""
    assert type(value) is dict and set(value) == {'corrupt', 'objects_zlib', 'metadata', 'calls'}
    assert len(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()) + 1 <= MAX_WIRE_BYTES
    assert type(value['objects_zlib']) is dict and len(value['objects_zlib']) <= 1
    objects = {}
    for key, encoded in value['objects_zlib'].items():
        assert type(key) is str and type(encoded) is str
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(encoded, validate=True), MAX_OBJECT_BYTES + 1)
        assert len(raw) <= MAX_OBJECT_BYTES and decoder.eof
        assert not decoder.unused_data and not decoder.unconsumed_tail
        objects[key] = base64.b64encode(raw).decode()
    return dict(corrupt=value['corrupt'], objects=objects,
                metadata=value['metadata'], calls=value['calls'])


class Cloud:
    def __init__(self, *, corrupt=False):
        self.objects, self.parts, self.metadata = {}, {}, {}
        self.corrupt = corrupt
        self.calls, self.bodies = [], []

    def head_object(self, **args):
        key = args['Key']
        self.calls.append('head')
        return dict(ContentLength=len(self.objects[key]), Metadata=self.metadata[key], ETag='"tiny-original"')

    def create_multipart_upload(self, **args):
        self.calls.append('create')
        self.parts[args['Key']] = []
        self.metadata[args['Key']] = args['Metadata']
        return dict(UploadId='tiny-owned-upload')

    def upload_part(self, **args):
        self.calls.append('part')
        self.parts[args['Key']].append(args['Body'])
        assert sum(map(len, self.parts[args['Key']])) <= MAX_OBJECT_BYTES
        return dict(ETag='"tiny-part"')

    def complete_multipart_upload(self, **args):
        self.calls.append('complete')
        self.objects[args['Key']] = b''.join(self.parts[args['Key']])
        return {}

    def abort_multipart_upload(self, **args):
        self.calls.append('abort')
        return {}

    def get_object(self, **args):
        self.calls.append('readback')
        body = io.BytesIO(self.objects[args['Key']] + (b'foreign' if self.corrupt else b''))
        self.bodies.append(body)
        return dict(Body=body)

    def close(self):
        self.calls.append('client_closed')
