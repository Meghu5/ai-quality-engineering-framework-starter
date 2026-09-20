from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityIdentityConfig:
    valid_user_token: str = "valid-user-token"
    another_user_token: str = "another-user-token"
    admin_token: str = "admin-test-token"
    invalid_token: str = "invalid-test-token"
    expired_token: str = "expired-test-token"


SECURITY_IDENTITIES = SecurityIdentityConfig()
