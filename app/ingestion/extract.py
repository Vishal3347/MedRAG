"""Page-by-page PDF extraction: PyMuPDF -> pdfplumber -> Tesseract OCR (each only when needed)."""
from __future__ import annotations

import importlib.util
import io
import logging
import shutil
from pathlib import Path
from typing import Iterator

from app.ingestion.models import Line, RawPage

log = logging.getLogger(__name__)


def ocr_available() -> bool:
    return bool(
        shutil.which("tesseract")
        and importlib.util.find_spec("pytesseract")
        and importlib.util.find_spec("PIL")
    )


def read_pdf_title(pdf_path: Path) -> str | None:
    import fitz

    with fitz.open(pdf_path) as doc:
        title = (doc.metadata or {}).get("title") or ""
    title = title.strip()
    return title if len(title) > 3 else None


def page_count(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def _fitz_page(page) -> RawPage:
    data = page.get_text("dict")
    lines: list[Line] = []
    for block_no, block in enumerate(data.get("blocks", [])):
        if block.get("type", 0) != 0:  # skip images
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text")]
            text = "".join(s["text"] for s in spans)
            if not text.strip():
                continue
            dominant = max(spans, key=lambda s: len(s["text"].strip()))
            font = str(dominant.get("font", "")).lower()
            bold = bool(dominant.get("flags", 0) & 16) or any(w in font for w in ("bold", "black", "heavy"))
            lines.append(
                Line(text=text, size=float(dominant.get("size", 0.0)), bold=bold, block=block_no, y=float(line["bbox"][1]))
            )
    return RawPage(number=page.number + 1, height=float(page.rect.height), lines=lines)


def _chars(lines: list[Line]) -> int:
    return sum(len(l.text.strip()) for l in lines)


class _Plumber:
    def __init__(self, path: Path):
        self.path = path
        self._pdf = None

    def lines(self, number: int) -> list[Line]:
        if self._pdf is None:
            import pdfplumber

            self._pdf = pdfplumber.open(self.path)
        text = self._pdf.pages[number - 1].extract_text() or ""
        return [Line(text=t, block=0) for t in text.splitlines() if t.strip()]

    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()


def _ocr_lines(page, lang: str, dpi: int = 200) -> list[Line]:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    text = pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))), lang=lang)
    lines: list[Line] = []
    block = 0
    for raw in text.splitlines():
        if not raw.strip():
            block += 1
        else:
            lines.append(Line(text=raw, block=block))
    return lines


def iter_raw_pages(
    pdf_path: Path,
    *,
    ocr: bool = True,
    ocr_lang: str = "eng",
    min_chars: int = 30,
    first_page: int = 1,
    last_page: int | None = None,
) -> Iterator[RawPage]:
    import fitz

    use_ocr = ocr and ocr_available()
    if ocr and not use_ocr:
        log.warning("OCR fallback requested but tesseract/pytesseract/Pillow are not installed; scanned pages will be empty")
    plumber = _Plumber(pdf_path)
    try:
        with fitz.open(pdf_path) as doc:
            last = min(last_page or doc.page_count, doc.page_count)
            for idx in range(max(first_page, 1) - 1, last):
                page = doc.load_page(idx)
                raw = _fitz_page(page)
                if _chars(raw.lines) < min_chars:
                    try:
                        alt = plumber.lines(raw.number)
                    except Exception as exc:  # noqa: BLE001
                        log.debug("pdfplumber fallback failed on page %d: %s", raw.number, exc)
                        alt = []
                    if _chars(alt) >= min_chars:
                        raw.lines, raw.source = alt, "plumber"
                if _chars(raw.lines) < min_chars and use_ocr:
                    try:
                        ocr_lines = _ocr_lines(page, ocr_lang)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("OCR failed on page %d: %s", raw.number, exc)
                        ocr_lines = []
                    if _chars(ocr_lines) >= min_chars:
                        raw.lines, raw.source = ocr_lines, "ocr"
                if _chars(raw.lines) < min_chars:
                    raw.source = "empty" if not raw.lines else raw.source
                yield raw
    finally:
        plumber.close()
