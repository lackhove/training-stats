"""Shared-secret bearer authentication for HTTP-based transports.

FastMCP has no built-in static-token server provider, so this module
implements the documented extension point (``TokenVerifier.verify_token``)
for a single shared secret. The secret is wrapped in :class:`SecretStr` and
never appears in logs or ``repr``.
"""

from __future__ import annotations

import hmac

from fastmcp.server.auth import AccessToken, TokenVerifier
from pydantic import SecretStr


class SharedSecretBearerVerifier(TokenVerifier):
    """Accepts ``Authorization: Bearer <token>`` matching a shared secret.

    Uses a constant-time comparison; empty tokens always fail so that clients
    configured with a missing key (e.g. Open WebUI's empty Bearer header) get
    a clean 401 instead of access.
    """

    def __init__(self, secret: SecretStr) -> None:
        super().__init__()
        self._secret = secret

    async def verify_token(self, token: str) -> AccessToken | None:
        expected = self._secret.get_secret_value()
        if not token or not hmac.compare_digest(token.encode(), expected.encode()):
            return None
        return AccessToken(
            token=token,
            client_id="bearer-client",
            scopes=[],
            expires_at=None,
        )

    def __repr__(self) -> str:
        return "SharedSecretBearerVerifier(secret=***)"
