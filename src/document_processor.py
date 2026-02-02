from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from docx import Document
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extract text from various document formats"""

    SUPPORTED_FORMATS = {'.pdf', '.docx', '.doc', '.txt'}

    @staticmethod
    def detect_format(file_path: Path) -> str:
        """Detect document format based on extension"""
        suffix = file_path.suffix.lower()
        format_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'doc',
            '.txt': 'txt'
        }
        return format_map.get(suffix, 'unknown')

    def extract_text_from_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF with page metadata"""
        logger.info(f"Processing PDF: {file_path.name}")
        pages = []

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                logger.info(f"Total pages: {total_pages:,}")

                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        text = page.extract_text()

                        if text and text.strip():  # Only add non-empty pages
                            pages.append({
                                'page_number': page_num,
                                'text': text,
                                'char_count': len(text),
                                'word_count': len(text.split())
                            })

                        if page_num % 100 == 0:
                            logger.info(
                                f"Processed {page_num:,}/{total_pages:,} pages ({page_num / total_pages * 100:.1f}%)")

                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        continue

                logger.info(f"✅ Extracted text from {len(pages):,} pages")

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

        return pages

    def extract_text_from_docx(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text from DOCX"""
        logger.info(f"Processing DOCX: {file_path.name}")

        try:
            doc = Document(file_path)

            paragraphs = []
            for idx, para in enumerate(doc.paragraphs, start=1):
                if para.text and para.text.strip():
                    paragraphs.append({
                        'paragraph_number': idx,
                        'text': para.text,
                        'char_count': len(para.text),
                        'word_count': len(para.text.split())
                    })

            logger.info(f"✅ Extracted {len(paragraphs):,} paragraphs")
            return paragraphs

        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            raise

    def extract_text_from_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text from TXT file"""
        logger.info(f"Processing TXT: {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            logger.info(f"✅ Extracted {len(text):,} characters")

            return [{
                'page_number': 1,
                'text': text,
                'char_count': len(text),
                'word_count': len(text.split())
            }]

        except Exception as e:
            logger.error(f"Error processing TXT: {e}")
            raise

    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process document based on format"""

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_format = self.detect_format(file_path)

        if doc_format not in ['pdf', 'docx', 'txt']:
            raise ValueError(f"Unsupported document format: {doc_format}. Supported: {self.SUPPORTED_FORMATS}")

        logger.info(f"Document format detected: {doc_format.upper()}")

        if doc_format == 'pdf':
            return self.extract_text_from_pdf(file_path)
        elif doc_format == 'docx':
            return self.extract_text_from_docx(file_path)
        elif doc_format == 'txt':
            return self.extract_text_from_txt(file_path)