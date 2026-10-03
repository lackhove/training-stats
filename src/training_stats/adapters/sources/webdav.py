"""WebDAV (Nextcloud) activity source built on httpx.

Implements directory listing via ``PROPFIND`` and file download via ``GET``.
Listing returns paths relative to the configured collection; downloads are
resolved against the same collection URL. Credentials are supplied by the
caller (resolved from environment variables) and are never logged.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal, NoReturn
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree

import httpx

from ...domain.models import SourceFile
from ...services.errors import SourceError
from ..config import WebDAVCredentials, WebDAVSourceConfig
from ..errors import AuthenticationError, RemoteFileError
from .base import ActivitySource, is_fit_file

logger = logging.getLogger(__name__)

_DAV_NS = "DAV:"

_PROPFIND_BODY = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<d:propfind xmlns:d="DAV:">'
    "<d:prop>"
    "<d:resourcetype/>"
    "<d:getcontentlength/>"
    "<d:getlastmodified/>"
    "</d:prop>"
    "</d:propfind>"
)


class WebDAVSource(ActivitySource):
    """Reads FIT files from a WebDAV collection such as Nextcloud."""

    def __init__(
        self,
        config: WebDAVSourceConfig,
        credentials: WebDAVCredentials,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = config.name
        self.config = config
        self._credentials = credentials
        self._transport = transport
        self._http_client: httpx.Client | None = None
        base = str(config.base_url).rstrip("/") + "/"
        trimmed_path = config.path.strip("/")
        self._collection_url = base + quote(trimmed_path) + "/" if trimmed_path else base
        self._collection_path = _url_path(self._collection_url)

    def list_files(self) -> list[SourceFile]:
        depth = "infinity" if self.config.recursive else "1"
        try:
            response = self._client().request(
                "PROPFIND",
                self._collection_url,
                content=_PROPFIND_BODY,
                headers={"Depth": depth, "Content-Type": "application/xml"},
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            msg = (
                f"webdav source {self.name!r}: request timed out while listing {self.config.path!r}"
            )
            raise SourceError(msg) from exc
        except httpx.HTTPStatusError as exc:
            self._raise_status(exc, operation="listing")
        except httpx.HTTPError as exc:
            msg = f"webdav source {self.name!r}: listing failed: {type(exc).__name__}"
            raise SourceError(msg) from exc
        return self._parse_propfind(response.text)

    def _raise_status(
        self,
        exc: httpx.HTTPStatusError,
        *,
        operation: Literal["listing", "download"],
        path: str = "",
    ) -> NoReturn:
        """Map an HTTP error status onto the matching source error.

        ``operation="listing"`` reports the configured collection; a ``path``
        names the file a download was requested for.
        """
        status = exc.response.status_code
        if status == 401:
            msg = f"webdav source {self.name!r}: authentication failed (HTTP 401)"
            raise AuthenticationError(msg) from exc
        if operation == "listing":
            if status == 404:
                msg = (
                    f"webdav source {self.name!r}: remote directory not found: {self.config.path!r}"
                )
                raise SourceError(msg) from exc
            msg = f"webdav source {self.name!r}: unexpected HTTP status {status} while listing"
        else:
            if status == 404:
                msg = f"webdav source {self.name!r}: remote file not found: {path}"
                raise RemoteFileError(msg) from exc
            msg = f"webdav source {self.name!r}: unexpected HTTP status {status} downloading {path}"
        raise RemoteFileError(msg) from exc

    def _parse_propfind(self, body: str) -> list[SourceFile]:
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            msg = f"webdav source {self.name!r}: malformed PROPFIND response"
            raise SourceError(msg) from exc
        files: list[SourceFile] = []
        for response_node in root.findall(f"{{{_DAV_NS}}}response"):
            href_node = response_node.find(f"{{{_DAV_NS}}}href")
            if href_node is None or href_node.text is None:
                continue
            item_path = _url_path(unquote(href_node.text))
            if not item_path.startswith(self._collection_path):
                continue
            relative = item_path[len(self._collection_path) :]
            if not relative or relative == self.config.path.strip("/"):
                continue
            collection = response_node.find(
                f"{{{_DAV_NS}}}propstat/{{{_DAV_NS}}}prop/{{{_DAV_NS}}}resourcetype/{{{_DAV_NS}}}collection"
            )
            if collection is not None:
                continue
            if not is_fit_file(relative):
                continue
            files.append(
                SourceFile(
                    source_name=self.name,
                    path=relative,
                    size=self._prop_int(response_node, "getcontentlength"),
                    modified_at=self._prop_datetime(response_node, "getlastmodified"),
                )
            )
        logger.debug("webdav source %r discovered %d FIT files", self.name, len(files))
        return files

    @staticmethod
    def _prop_int(node: ElementTree.Element, prop: str) -> int | None:
        value = node.findtext(f"{{{_DAV_NS}}}propstat/{{{_DAV_NS}}}prop/{{{_DAV_NS}}}{prop}")
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _prop_datetime(node: ElementTree.Element, prop: str) -> datetime | None:
        value = node.findtext(f"{{{_DAV_NS}}}propstat/{{{_DAV_NS}}}prop/{{{_DAV_NS}}}{prop}")
        if value is None:
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    def read_file(self, source_file: SourceFile) -> bytes:
        url = self._collection_url + quote(source_file.path)
        try:
            response = self._client().get(url)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            msg = f"webdav source {self.name!r}: download timed out: {source_file.path}"
            raise RemoteFileError(msg) from exc
        except httpx.HTTPStatusError as exc:
            self._raise_status(exc, operation="download", path=source_file.path)
        except httpx.HTTPError as exc:
            msg = f"webdav source {self.name!r}: download failed: {type(exc).__name__}"
            raise RemoteFileError(msg) from exc
        return response.content

    def _client(self) -> httpx.Client:
        # One client per source so connections are reused across requests.
        if self._http_client is None:
            self._http_client = httpx.Client(
                auth=(self._credentials.username, self._credentials.password),
                verify=self.config.verify_tls,
                timeout=self.config.timeout_seconds,
                transport=self._transport,
            )
        return self._http_client


def _url_path(url: str) -> str:
    return urlsplit(url).path
