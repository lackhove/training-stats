"""Shared-secret bearer authentication for the REST server.

The token never appears in logs or ``repr``; comparison is constant-time.
Applied to the data routes only — ``/openapi.json`` and ``/docs`` stay
reachable so clients can discover the schema.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

bearer_scheme = HTTPBearer(auto_error=False, description="Shared-secret bearer token")


class BearerGuard:
    """FastAPI dependency enforcing one shared-secret bearer token."""

    def __init__(self, token: SecretStr) -> None:
        self._token = token

    def __repr__(self) -> str:
        return "BearerGuard(token=***)"

    async def __call__(
        self,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    ) -> None:
        provided = credentials.credentials if credentials is not None else ""
        expected = self._token.get_secret_value()
        if not provided or not secrets.compare_digest(
            provided.encode("utf-8"), expected.encode("utf-8")
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
