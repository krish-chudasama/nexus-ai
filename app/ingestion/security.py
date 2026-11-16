"""Upload validation: filename sanitisation, extension allow-list, size limit."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath

from app.core.errors import FileTooLarge, UnsupportedFileType, ValidationFailed

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
_UNSAFE = re.compile(r"[^A-Za-z0-9._\- ]+")


def safe_filename(name: str) -> str:
    """Strip any directory components and unsafe characters; never returns an empty name."""
    base = PureWindowsPath(PurePosixPath(name or "").name).name  # handles / and \ separators
    base = _UNSAFE.sub("_", base).strip(" .")
    base = re.sub(r"_+", "_", base)
    if not base:
        base = "file"
    stem, dot, ext = base.rpartition(".")
    if dot and len(stem) > 120:
        base = f"{stem[:120]}.{ext}"
    return base[:150]


def file_extension(filename: str) -> str:
    ext = PurePosixPath(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{ext or 'none'}'. Allowed: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )
    return ext


def validate_upload(filename: str, data: bytes, max_mb: int) -> tuple[str, str]:
    """Returns (safe_filename, extension). Raises on invalid uploads."""
    if not data:
        raise ValidationFailed("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise FileTooLarge(f"File exceeds the {max_mb} MB limit.")
    name = safe_filename(filename)
    ext = file_extension(name)
    # Cheap magic-byte checks so a renamed binary is not parsed as the wrong type.
    if ext == ".pdf" and not data.lstrip()[:5] == b"%PDF-":
        raise ValidationFailed("File content is not a valid PDF.")
    if ext == ".docx" and not data[:2] == b"PK":
        raise ValidationFailed("File content is not a valid DOCX.")
    if ext in {".txt", ".md", ".markdown"} and b"\x00" in data[:4096]:
        raise ValidationFailed("File appears to be binary, not text.")
    return name, ext
