import pdfplumber
import pytesseract
from PIL import Image
import io
from typing import List

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                for img in page.images:
                    pil_img = Image.open(io.BytesIO(img["stream"].get_data()))
                    ocr_text = pytesseract.image_to_string(pil_img)
                    text += ocr_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks