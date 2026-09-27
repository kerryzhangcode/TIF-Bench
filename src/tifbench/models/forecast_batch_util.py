"""Helpers for resume-safe forecast batch chunk directories."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def chunk_dir_for_custom_ids(batch_dir: Path, custom_ids: list[str]) -> Path:
    """Directory name is a fingerprint of the id set, not a pending-list offset.

    Resume after partial cache hits must not reuse another slice's batch artifacts.
    """
    digest = hashlib.sha256(
        "\n".join(sorted(custom_ids)).encode("utf-8")
    ).hexdigest()[:16]
    return batch_dir / f"chunk_{digest}"


def require_batch_mapping_keys(
    prior: dict[str, Any],
    custom_ids: list[str],
    *,
    chunk_dir: Path,
) -> None:
    missing = [cid for cid in custom_ids if cid not in prior]
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise RuntimeError(
            f"Batch output under {chunk_dir} missing custom_id(s): {preview}{more}"
        )
