"""Tests for the WebDAV source using mocked HTTP responses."""

from __future__ import annotations

import httpx
import pytest

from training_stats.adapters.config import WebDAVCredentials, WebDAVSourceConfig
from training_stats.adapters.errors import AuthenticationError, RemoteFileError
from training_stats.adapters.sources.webdav import WebDAVSource
from training_stats.services.errors import SourceError

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


PROPFIND_RESPONSE_SPACE_PATH = """<?xml version="1.0" encoding="utf-8"?>
<d:multistatus xmlns:d="DAV:">
  <d:response>
    <d:href>/remote.php/dav/files/alice/Garmin/My%20Rides/</d:href>
    <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop>
    <d:status>HTTP/1.1 200 OK</d:status></d:propstat>
  </d:response>
  <d:response>
    <d:href>/remote.php/dav/files/alice/Garmin/My%20Rides/ride%20one.FIT</d:href>
    <d:propstat><d:prop>
      <d:resourcetype/>
      <d:getcontentlength>4321</d:getcontentlength>
      <d:getlastmodified>Tue, 12 Aug 2026 08:12:31 GMT</d:getlastmodified>
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


def test_collection_path_with_space_matches_encoded_propfind_hrefs() -> None:
    """The quoted collection URL and percent-encoded hrefs compare canonically."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "PROPFIND":
            return httpx.Response(207, text=PROPFIND_RESPONSE_SPACE_PATH)
        return httpx.Response(200, content=b"\x00\x01fit-bytes")

    source = make_source(handler, make_config(path="Garmin/My Rides"))
    files = source.list_files()
    assert [item.path for item in files] == ["ride one.FIT"]
    assert files[0].size == 4321

    propfind = seen[0]
    assert propfind.url.raw_path == b"/remote.php/dav/files/alice/Garmin/My%20Rides/"

    source.read_file(files[0])
    download = seen[-1]
    assert download.url.raw_path == b"/remote.php/dav/files/alice/Garmin/My%20Rides/ride%20one.FIT"


def test_client_reused_across_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """PROPFIND + GET + a second listing share one httpx.Client."""
    import training_stats.adapters.sources.webdav as webdav_module

    created: list[int] = []
    real_client = httpx.Client

    def counting_client(*args: object, **kwargs: object) -> httpx.Client:
        created.append(1)
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(webdav_module.httpx, "Client", counting_client)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PROPFIND":
            return httpx.Response(207, text=PROPFIND_RESPONSE)
        return httpx.Response(200, content=b"\x00\x01fit-bytes")

    source = make_source(handler)
    files = source.list_files()
    source.read_file(files[0])
    source.list_files()
    assert len(created) == 1


def test_authentication_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    source = make_source(handler)
    with pytest.raises(AuthenticationError) as exc_info:
        source.list_files()
    assert str(exc_info.value) == "webdav source 'nextcloud': authentication failed (HTTP 401)"


def test_download_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PROPFIND":
            return httpx.Response(207, text=PROPFIND_RESPONSE)
        return httpx.Response(404)

    source = make_source(handler)
    with pytest.raises(RemoteFileError) as exc_info:
        source.read_file(source.list_files()[0])
    assert str(exc_info.value) == "webdav source 'nextcloud': remote file not found: ride1.FIT"


def test_directory_404() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    source = make_source(handler)
    with pytest.raises(SourceError) as exc_info:
        source.list_files()
    assert (
        str(exc_info.value)
        == "webdav source 'nextcloud': remote directory not found: 'Garmin/Activities'"
    )


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
    with pytest.raises(RemoteFileError) as exc_info:
        source.list_files()
    assert (
        str(exc_info.value) == "webdav source 'nextcloud': unexpected HTTP status 500 while listing"
    )


def test_unlisted_status_is_covered_by_raise_for_status() -> None:
    """Any 4xx/5xx (e.g. 403) maps to a domain error without a dedicated branch."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    source = make_source(handler)
    with pytest.raises(RemoteFileError, match="403"):
        source.list_files()


def test_download_unlisted_status_is_covered_by_raise_for_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PROPFIND":
            return httpx.Response(207, text=PROPFIND_RESPONSE)
        return httpx.Response(403)

    source = make_source(handler)
    with pytest.raises(RemoteFileError) as exc_info:
        source.read_file(source.list_files()[0])
    assert (
        str(exc_info.value)
        == "webdav source 'nextcloud': unexpected HTTP status 403 downloading ride1.FIT"
    )


def test_download_authentication_failure() -> None:
    """A 401 on download maps to the same authentication error as on listing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PROPFIND":
            return httpx.Response(207, text=PROPFIND_RESPONSE)
        return httpx.Response(401)

    source = make_source(handler)
    with pytest.raises(AuthenticationError) as exc_info:
        source.read_file(source.list_files()[0])
    assert str(exc_info.value) == "webdav source 'nextcloud': authentication failed (HTTP 401)"


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
