"""
Better Auth session-token verification for the FastAPI backend.

Identity is derived ONLY from a cryptographically verified Better Auth JWT
(signed by the Next.js app and exposed via its JWKS endpoint) — never from a
client-supplied user_id. This closes the IDOR hole where any caller could
read/delete another user's chats by passing their id.
"""
import os
from typing import Optional

import jwt
from jwt import PyJWKClient
from fastapi import Header, HTTPException
from dotenv import load_dotenv

load_dotenv()

# Base URL of the Next.js app that issues tokens (Better Auth `baseURL`).
BETTER_AUTH_URL = (os.getenv("BETTER_AUTH_URL") or "").rstrip("/")
# JWKS published by the Better Auth `jwt` plugin.
JWKS_URL = os.getenv("BETTER_AUTH_JWKS_URL") or (
    f"{BETTER_AUTH_URL}/api/auth/jwks" if BETTER_AUTH_URL else None
)

# Better Auth signs with EdDSA (Ed25519) by default; allow common alternates.
ALGORITHMS = ["EdDSA", "ES256", "RS256"]

_jwk_client: Optional[PyJWKClient] = PyJWKClient(JWKS_URL) if JWKS_URL else None


def auth_configured() -> bool:
    return _jwk_client is not None


def _verify_token(token: str) -> str:
    """Verify a Better Auth JWT and return the user id (sub). Raises on failure."""
    if not _jwk_client:
        raise HTTPException(status_code=503, detail="Authentication not configured")
    try:
        signing_key = _jwk_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=ALGORITHMS,
            issuer=BETTER_AUTH_URL or None,
            options={
                "verify_aud": False,
                "verify_iss": bool(BETTER_AUTH_URL),
                "require": ["exp", "sub"],
            },
            leeway=10,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing subject")
    return sub


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    return None


def get_current_user_id(authorization: Optional[str] = Header(default=None)) -> str:
    """Dependency: require a valid signed-in user. 401 otherwise."""
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    return _verify_token(token)


def get_optional_user_id(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    """Dependency: return verified user id if present, else None (guest)."""
    token = _extract_bearer(authorization)
    if not token:
        return None
    return _verify_token(token)
