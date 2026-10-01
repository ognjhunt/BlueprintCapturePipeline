from types import SimpleNamespace
import pytest
from blueprint_pipeline.s3_compatible_transport import s3_compatible_client


@pytest.mark.parametrize("endpoint", [None, "", "http://spaces.example", "https://s3.amazonaws.com", "https://s3.us-east-1.amazonaws.com", "https://bucket.s3.cn-north-1.amazonaws.com.cn", "https://s3.us-east-1.api.aws"])
def test_missing_or_aws_endpoint_refuses_before_sdk(endpoint):
    def forbidden(*args, **kwargs):
        raise AssertionError("SDK invoked")
    with pytest.raises(ValueError):
        s3_compatible_client(SimpleNamespace(client=forbidden), endpoint_url=endpoint, aws_access_key_id="key", aws_secret_access_key="secret")


@pytest.mark.parametrize("endpoint", ["https://s3.us-west-004.backblazeb2.com", "https://nyc3.digitaloceanspaces.com", "https://account.r2.cloudflarestorage.com", "https://s3api-eur-is-1.runpod.io"])
def test_compatible_transport_preserves_explicit_sdk_parameters(endpoint):
    observed = []
    sdk = SimpleNamespace(client=lambda *args, **kwargs: observed.append((args, kwargs)) or "client")
    assert s3_compatible_client(sdk, endpoint_url=endpoint, aws_access_key_id="key", aws_secret_access_key="secret", region_name="region") == "client"
    assert observed == [(('s3',), dict(endpoint_url=endpoint, aws_access_key_id="key", aws_secret_access_key="secret", region_name="region"))]


def test_no_ambient_credential_discovery():
    with pytest.raises(ValueError, match="explicit_credentials"):
        s3_compatible_client(None, endpoint_url="https://nyc3.digitaloceanspaces.com")


@pytest.mark.parametrize("mode", [0o644, 0o666, 0o600])
def test_private_credential_permissions(tmp_path, mode):
    from blueprint_pipeline.s3_compatible_transport import read_private_credential
    path = tmp_path / "credential"
    path.write_text("fixture-secret\n")
    path.chmod(mode)
    if mode == 0o600:
        assert read_private_credential(path) == "fixture-secret"
    else:
        with pytest.raises(ValueError, match="private_credential_file"):
            read_private_credential(path)


def test_private_credential_rejects_symlink(tmp_path):
    from blueprint_pipeline.s3_compatible_transport import read_private_credential
    target = tmp_path / "credential"
    target.write_text("fixture-secret")
    target.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        read_private_credential(link)
