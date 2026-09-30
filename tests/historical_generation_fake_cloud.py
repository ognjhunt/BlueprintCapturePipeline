"""Tiny in-process object transport; never contacts a provider or credentials."""
import io


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
        assert sum(map(len, self.parts[args['Key']])) <= 65536
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
