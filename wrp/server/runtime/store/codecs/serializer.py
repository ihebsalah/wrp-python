# wrp/server/runtime/store/codecs/serializer.py
from __future__ import annotations

import json
from typing import Any


def dumps(obj: Any) -> bytes:
    # stable JSON without whitespace bloat
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def loads(buf: bytes) -> Any:
    return json.loads(buf.decode("utf-8"))
