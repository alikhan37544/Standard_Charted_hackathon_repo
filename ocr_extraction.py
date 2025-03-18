import cv2
import pytesseract
import re
from paddleocr import PaddleOCR
import spacy

ocr = PaddleOCR(use_angle_cls=True, lang='en')
nlp = spacy.load("en_core_web_sm")
AADHAAR_PATTERN = r"\b\d{4}\s\d{4}\s\d{4}\b"

def extract_aadhaar_details(text):
    """Extract Aadhaar Number and Name from OCR text."""
    aadhaar_match = re.search(AADHAAR_PATTERN, text)
    aadhaar_number = aadhaar_match.group() if aadhaar_match else "Not Found"

    doc = nlp(text)
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Not Found")

    return aadhaar_number, name

def process_aadhaar_card(image_path):
    """Processes an Aadhaar card image for OCR extraction."""
    text = " ".join([line[1][0] for line in ocr.ocr(image_path, cls=True)[0]]) or pytesseract.image_to_string(cv2.imread(image_path))
    return extract_aadhaar_details(text)

if __name__ == "__main__":
    image_path = "sample_aadhaar.jpg"
    print(process_aadhaar_card(image_path))
