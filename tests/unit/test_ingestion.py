import io

import docx
import pymupdf
import pytest

from app.core.errors import (
    DuplicateDocument,
    ExtractionError,
    FileTooLarge,
    UnsupportedFileType,
    ValidationFailed,
)
from app.ingestion.chunking import split_text
from app.ingestion.extractors import clean_text, extract
from app.ingestion.security import safe_filename, validate_upload


class TestSecurity:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("../../etc/passwd", "passwd"),
            ("C:\\Users\\x\\report.pdf", "report.pdf"),
            ("my file (1).txt", "my file _1_.txt"),
            ("", "file"),
            (".hidden", "hidden"),
        ],
    )
    def test_safe_filename(self, raw, expected):
        assert safe_filename(raw) == expected

    def test_rejects_bad_extension(self):
        with pytest.raises(UnsupportedFileType):
            validate_upload("evil.exe", b"MZ", 5)

    def test_rejects_empty_and_oversized(self):
        with pytest.raises(ValidationFailed):
            validate_upload("a.txt", b"", 5)
        with pytest.raises(FileTooLarge):
            validate_upload("a.txt", b"x" * (2 * 1024 * 1024), 1)

    def test_rejects_fake_pdf(self):
        with pytest.raises(ValidationFailed):
            validate_upload("a.pdf", b"not a pdf", 5)


class TestChunking:
    def test_short_text_single_chunk(self):
        assert split_text("hello world", 100, 10) == ["hello world"]

    def test_chunks_respect_size_and_overlap(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = split_text(text, 200, 40)
        assert len(chunks) > 3
        assert all(len(c) <= 200 for c in chunks)
        # overlap: consecutive chunks share some text
        assert chunks[0][-20:].split()[-1] in chunks[1]

    def test_all_content_preserved(self):
        words = [f"w{i}" for i in range(300)]
        chunks = split_text(" ".join(words), 150, 30)
        joined = " ".join(chunks)
        assert all(w in joined for w in words)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            split_text("abc", 10, 10)


class TestExtraction:
    def test_clean_text(self):
        assert clean_text("a \x00 b\r\n\r\n\r\n\r\nc") == "a b\n\nc"

    def test_txt_and_md(self):
        doc = extract(b"# Title\n\nBody text", ".md")
        assert "Body text" in doc.full_text

    def test_pdf_with_pages(self):
        pdf = pymupdf.open()
        for t in ("First page content", "Second page content"):
            pdf.new_page().insert_text((72, 72), t)
        data = pdf.tobytes()
        doc = extract(data, ".pdf")
        assert doc.num_pages == 2
        assert [s.page for s in doc.segments] == [1, 2]

    def test_docx(self):
        d = docx.Document()
        d.add_paragraph("Paragraph one about elephants.")
        t = d.add_table(rows=1, cols=2)
        t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Cell A", "Cell B"
        buf = io.BytesIO()
        d.save(buf)
        text = extract(buf.getvalue(), ".docx").full_text
        assert "elephants" in text and "Cell A | Cell B" in text

    def test_empty_pdf_raises(self):
        pdf = pymupdf.open()
        pdf.new_page()
        with pytest.raises(ExtractionError):
            extract(pdf.tobytes(), ".pdf")


class TestDocumentService:
    def test_ingest_list_delete(self, services):
        info = services.documents.ingest("notes.txt", b"Alpha beta gamma. " * 100)
        assert info.num_chunks >= 1 and info.status == "indexed"
        assert services.documents.list().total_chunks == info.num_chunks
        assert services.store.count_for(info.id) == info.num_chunks
        services.documents.delete(info.id)
        assert services.documents.list().documents == []
        assert services.store.count() == 0
        assert not list(services.settings.uploads_dir.glob(f"{info.id}.*"))

    def test_duplicate_detection(self, services):
        services.documents.ingest("a.txt", b"same content here")
        with pytest.raises(DuplicateDocument):
            services.documents.ingest("b.txt", b"same content here")

    def test_registry_persists(self, services, settings):
        from app.embeddings.embedders import HashingEmbedder
        from app.services.container import build_services

        services.documents.ingest("a.txt", b"persist me please")
        again = build_services(settings, HashingEmbedder())
        assert len(again.documents.list().documents) == 1
