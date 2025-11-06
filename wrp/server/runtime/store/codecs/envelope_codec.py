# wrp/server/runtime/store/codecs/envelope_codec.py
from __future__ import annotations

from typing import Any, Optional

from . import serializer
from .crypto import decrypt_aes_gcm, encrypt_aes_gcm


def pack(obj: Any, *, key: bytes | None) -> bytes:
    """
    Returns bytes of an envelope.
    - If key is None: {"v":1,"alg":"none","ct":<json bytes>}
    - If key present: {"v":1,"alg":"aes-gcm","nonce":...,"ct":...}
    """
    raw = serializer.dumps(obj)
    if not key:
        env = {"v": 1, "alg": "none", "ct": raw.decode("utf-8")}
        return serializer.dumps(env)

    nonce, ct = encrypt_aes_gcm(raw, key)
    env = {"v": 1, "alg": "aes-gcm", "nonce": nonce.hex(), "ct": ct.hex()}
    return serializer.dumps(env)


def unpack(buf: bytes, *, key: Optional[bytes]) -> Any:
    env = serializer.loads(buf)
    if not isinstance(env, dict) or env.get("v") != 1:
        # legacy/raw (defensive)
        return env
    alg = env.get("alg")
    if alg == "none":
        return serializer.loads(env["ct"].encode("utf-8"))
    if alg == "aes-gcm":
        nonce = bytes.fromhex(env["nonce"])
        ct = bytes.fromhex(env["ct"])
        try:
            # Use a dummy key if none is provided, causing a predictable failure.
            raw = decrypt_aes_gcm(nonce, ct, key or b"\x00" * 32)
        except Exception as e:  # wrong key / corrupted data
            raise ValueError("Envelope decrypt failed: wrong key or corrupted data") from e
        return serializer.loads(raw)
    # unknown alg, pass-through
    return env