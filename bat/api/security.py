"""Authentication: credential to :class:`TenantContext`.

This is the only place a tenant identity is minted. Everything downstream
receives an already-resolved context and never re-parses a header, so there is
exactly one code path that decides who a caller is.

Design notes
------------
* Keys are compared by SHA-256 digest with :func:`hmac.compare_digest`, so a
  timing measurement cannot recover a key byte by byte.
* Only digests are configured, so a leaked config file does not yield usable
  credentials.
* A caller may not choose its own tenant. ``X-Tenant-Id``, if present, is
  treated as an assertion to be checked against the credential, never as the
  source of tenancy.

For production this registry moves behind a database with per-key revocation,
rotation and last-used tracking; the lookup surface stays
``digest -> ApiKeyRecord``, so only :class:`ApiKeyAuthenticator` changes.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Iterable

from fastapi import Request

from bat.domain.conversation import utcnow
from bat.domain.errors import AuthenticationError, AuthorizationError
from bat.domain.tenancy import Principal, Scope, TenantContext
from bat.settings import ApiKeyRecord

API_KEY_HEADER = "X-API-Key"
TENANT_HEADER = "X-Tenant-Id"
_BEARER_PREFIX = "bearer "
#: Public key prefix, so leaked keys are greppable in logs and repos.
KEY_PREFIX = "bat_sk_"


def generate_api_key() -> tuple[str, str]:
    """Return ``(plaintext, sha256_digest)``. Store only the digest."""
    plaintext = f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"
    return plaintext, hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def extract_credential(request: Request) -> str | None:
    """Pull the raw key from either supported header, preferring Bearer."""
    authorization = request.headers.get("authorization")
    if authorization and authorization.lower().startswith(_BEARER_PREFIX):
        candidate = authorization[len(_BEARER_PREFIX) :].strip()
        if candidate:
            return candidate
    header_key = request.headers.get(API_KEY_HEADER)
    return header_key.strip() if header_key else None


class ApiKeyAuthenticator:
    """Resolves an API key to a :class:`TenantContext`."""

    __slots__ = ("_by_digest",)

    def __init__(self, records: Iterable[ApiKeyRecord]) -> None:
        self._by_digest: dict[str, ApiKeyRecord] = {r.key_sha256: r for r in records}

    def __len__(self) -> int:
        return len(self._by_digest)

    def _lookup(self, plaintext: str) -> ApiKeyRecord | None:
        """Constant-time digest comparison across the whole registry.

        Every candidate is compared even after a match so that lookup time does
        not depend on the key's position in the registry.
        """
        digest = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
        found: ApiKeyRecord | None = None
        for candidate_digest, record in self._by_digest.items():
            if hmac.compare_digest(digest, candidate_digest):
                found = record
        return found

    def authenticate(self, request: Request, *, request_id: str) -> TenantContext:
        credential = extract_credential(request)
        if not credential:
            raise AuthenticationError(
                "missing credential; supply an Authorization: Bearer header "
                f"or {API_KEY_HEADER}"
            )

        record = self._lookup(credential)
        if record is None:
            # Identical message and status for unknown vs. malformed keys, so a
            # caller cannot enumerate valid key shapes from the response.
            raise AuthenticationError("invalid credential")

        asserted_tenant = request.headers.get(TENANT_HEADER)
        if asserted_tenant and asserted_tenant != record.tenant_id:
            raise AuthorizationError(
                "credential is not valid for the requested tenant",
                details={"tenant_id": asserted_tenant},
            )

        principal = Principal(
            id=record.principal_id,
            tenant_id=record.tenant_id,
            scopes=frozenset(record.scopes),
            display_name=record.label,
        )
        return TenantContext(
            tenant_id=record.tenant_id,
            principal=principal,
            request_id=request_id,
            started_at=utcnow(),
        )


def require_scopes(context: TenantContext, *scopes: Scope) -> TenantContext:
    """Thin wrapper so routes read declaratively."""
    return context.require(*scopes)
