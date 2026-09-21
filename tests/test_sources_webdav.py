"""Tests for the WebDAV source using mocked HTTP responses."""

from __future__ import annotations

import httpx
import pytest

from training_stats.config import WebDAVCredentials, WebDAVSourceConfig
from training_stats.errors import AuthenticationError, RemoteFileError, SourceError
from training_stats.sources.webdav import WebDAVSource

BASE_URL = "https://cloud.example.com/remote.php/dav/files/alice/"

PROPFIND_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<d:multistatus xmlns:d="DAV:">
  <d:response>
    <d:href>/remote.php/dav/files/alice/Garmin/Activities/</d:href>
    <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop>
    <d:status>HTTP/1.1 200 OK</d:status></d:propstat>
  </d:response>
  <d:response>
    <d:href>/remote.php/dav/files/alice/Garmin/Activities/ride1.FIT</d:href>
    <d:propstat><d:prop>
      <d:resourcetype/>
      <d:getcontentlength>1234</d:getcontentlength>
      <d:getlastmodified>Tue, 12 Aug 2026 08:12:31 GMT</d:getlastmodified>
    </d:prop>
    <d:status>HTTP/1.1 200 OK</d:status></d:propstat>
  </d:response>
  <d:response>
    <d:href>/remote.php/dav/files/alice/Garmin/Activities/notes.txt</d:href>
    <d:propstat><d:prop>
      <d:resourcetype/>
      <d:getcontentlength>10</d:getcontentlength>
    </d:prop>
    <d:status>HTTP/1.1 200 OK</d:status></d:propstat>
  </d:response>
  <d:response>
    <d:href>/remote.php/dav/files/alice/Garmin/Activities/sub/</d:href>
    <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop>
    <d:status>HTTP/1.1 200 OK</d:status></d:propstat>
  </d:response>
  <d:response>
    <d:href>/remote.php/dav/files/alice/Garmin/Activities/sub/ride2.fit</d:href>
    <d:propstat><d:prop>
      <d:resourcetype/>
      <d:getcontentlength>2000</d:getcontentlength>
    </d:prop>
    <d:status>HTTP/1.1 200 OK</d:status></d:propstat>
  </d:response>
  <d:response>
    <d:href>/remote.php/dav/files/alice/Other/place.fit</d:href>
    <d:propstat><d:prop>
      <d:resourcetype/>
      <d:getcontentlength>9999</d:getcontentlength>
    </d:prop>
    <d:status>HTTP/1.1 200 OK</d:status></d:propstat>
  </d:response>
</d:multistatus>
"""


def make_config(**overrides: object) -> WebDAVSourceConfig:
    values: dict[str, object] = {
        "name": "nextcloud",
        "base_url": BASE_URL,
        "path": "Garmin/Activities",
        "username_env": "NC_USER",
        "password_env": "NC_PASS",
    }
    values.update(overrides)
    return WebDAVSourceConfig(**values)  # type: ignore[arg-type]


def make_source(
    handler: httpx.MockTransport.Handler, config: WebDAVSourceConfig | None = None
) -> WebDAVSource:
    from pydantic import SecretStr

    credentials = WebDAVCredentials(username=SecretStr("alice"), password=SecretStr("secret"))
    return WebDAVSource(
        config or make_config(), credentials, transport=httpx.MockTransport(handler)
    )


def test_successful_listing() -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(207, text=PROPFIND_RESPONSE)

    source = make_source(handler)
    files = source.list_files()
    paths = [item.path for item in files]
    assert paths == ["ride1.FIT", "sub/ride2.fit"]
    assert files[0].size == 1234
    assert files[0].modified_at is not None
    request = seen_requests[0]
    assert request.method == "PROPFIND"
    assert request.headers["Depth"] == "infinity"
    assert "authorization" in request.headers


def test_successful_download() -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        if request.method == "PROPFIND":
            return httpx.Response(207, text=PROPFIND_RESPONSE)
        return httpx.Response(200, content=b"\x00\x01fit-bytes")

    source = make_source(handler)
    content = source.read_file(source.list_files()[0])
    assert content == b"\x00\x01fit-bytes"
    # The download URL must include the configured collection path.
    download = seen_requests[-1]
    assert download.url.path == "/remote.php/dav/files/alice/Garmin/Activities/ride1.FIT"

    nested = source.read_file(source.list_files()[1])
    assert nested == b"\x00\x01fit-bytes"
    assert (
        seen_requests[-1].url.path == "/remote.php/dav/files/alice/Garmin/Activities/sub/ride2.fit"
    )


def test_authentication_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    source = make_source(handler)
    with pytest.raises(AuthenticationError, match="401"):
        source.list_files()


def test_download_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PROPFIND":
            return httpx.Response(207, text=PROPFIND_RESPONSE)
        return httpx.Response(404)

    source = make_source(handler)
    with pytest.raises(RemoteFileError, match="not found"):
        source.read_file(source.list_files()[0])


def test_directory_404() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    source = make_source(handler)
    with pytest.raises(SourceError, match="not found"):
        source.list_files()


def test_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=httpx.Request("PROPFIND", BASE_URL))

    source = make_source(handler)
    with pytest.raises(SourceError, match="timed out"):
        source.list_files()


def test_download_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PROPFIND":
            return httpx.Response(207, text=PROPFIND_RESPONSE)
        raise httpx.ReadTimeout("too slow", request=request)

    source = make_source(handler)
    with pytest.raises(RemoteFileError, match="timed out"):
        source.read_file(source.list_files()[0])


def test_malformed_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, text="<not-xml")

    source = make_source(handler)
    with pytest.raises(SourceError, match="malformed"):
        source.list_files()


def test_unexpected_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    source = make_source(handler)
    with pytest.raises(RemoteFileError, match="500"):
        source.list_files()


def test_non_recursive_uses_depth_1() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Depth"])
        return httpx.Response(207, text=PROPFIND_RESPONSE)

    source = make_source(handler, make_config(recursive=False))
    source.list_files()
    assert seen == ["1"]


def test_no_toplevel_path_listing() -> None:
    """WebDAV roots without a sub-path still work."""
    from pydantic import SecretStr

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, text=PROPFIND_RESPONSE)

    credentials = WebDAVCredentials(username=SecretStr("alice"), password=SecretStr("secret"))
    source = WebDAVSource(make_config(path=""), credentials, transport=httpx.MockTransport(handler))
    files = source.list_files()
    assert len(files) == 3
