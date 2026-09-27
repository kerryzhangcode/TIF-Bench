"""Path helpers and simple CSV/JSON/JSONL IO. No business rules."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, TypeVar, get_origin

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON via temp file + os.replace so partial writes cannot corrupt the target."""
    ensure_dir(path.parent)
    payload = json.dumps(data, indent=2, default=str) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def write_models_csv(path: Path, models: list[T], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    rows = [m.model_dump(mode="json") for m in models]
    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (list, dict))
                             else row.get(k, "") for k in fieldnames})


def read_models_csv(path: Path, model_cls: type[T]) -> list[T]:
    if not path.exists():
        return []
    # Only blank→None for fields whose annotation explicitly allows None
    nullable_fields: set[str] = set()
    json_fields: set[str] = set()
    for name, field in model_cls.model_fields.items():
        ann = field.annotation
        # Optional[X] / X | None
        origin = getattr(ann, "__args__", None)
        if origin and type(None) in origin:
            nullable_fields.add(name)
        if get_origin(ann) in (list, dict):
            json_fields.add(name)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: list[T] = []
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v == "" and k in nullable_fields:
                    cleaned[k] = None
                elif k in json_fields and v:
                    cleaned[k] = json.loads(v)
                else:
                    cleaned[k] = v
            out.append(model_cls.model_validate(cleaned))
        return out


def project_root() -> Path:
    # Editable installs retain the checkout path. A wheel install resolves the
    # package in site-packages, so use the current checkout in that case.
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "config" / "default.yaml").is_file():
        return source_root
    cwd = Path.cwd()
    if (cwd / "config" / "default.yaml").is_file():
        return cwd
    raise RuntimeError("Run tifbench from the repository root containing config/default.yaml")
