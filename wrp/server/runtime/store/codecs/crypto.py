# wrp/server/runtime/store/codecs/crypto.py
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover
    AESGCM = None  # type: ignore


def _rand_bytes(n: int) -> bytes:
    return os.urandom(n)


def load_or_create_local_key(path: str | Path) -> bytes:
    """
    Load a 256-bit key from path; create if missing (chmod 0600).
    File stores base64 key.
    """
    p = Path(path)
    if not p.exists():
        key = os.urandom(32)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(base64.b64encode(key))
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass
        return key
    with open(p, "rb") as f:
        data = f.read().strip()
    try:
        return base64.b64decode(data)
    except Exception as e:  # pragma: no cover
        raise ValueError("Invalid key file contents") from e


def load_key_from_env(var: str) -> Optional[bytes]:
    v = os.getenv(var)
    if not v:
        return None
    return base64.b64decode(v)


def encrypt_aes_gcm(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    """
    Returns (nonce, ciphertext). 12-byte nonce.
    """
    if AESGCM is None:  # pragma: no cover
        # cryptography not installed → best-effort fallback (no encryption)
        return b"", plaintext
    nonce = _rand_bytes(12)
    ct = AESGCM(key).encrypt(nonce, plaintext, None)
    return nonce, ct


def decrypt_aes_gcm(nonce: bytes, ciphertext: bytes, key: bytes) -> bytes:
    if AESGCM is None:  # pragma: no cover
        return ciphertext
    return AESGCM(key).decrypt(nonce, ciphertext, None)
