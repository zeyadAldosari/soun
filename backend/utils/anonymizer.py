import pydicom
import cv2
import numpy as np
import easyocr
from pydicom.uid import generate_uid
import re

SENSITIVE_TAGS = [
    (0x0010, 0x0010), (0x0010, 0x0020), (0x0010, 0x0030), (0x0010, 0x0040),
    (0x0008, 0x0080), (0x0008, 0x0090), (0x0008, 0x1010), (0x0008, 0x1030),
    (0x0008, 0x1048), (0x0008, 0x0050), (0x0010, 0x1000), (0x0010, 0x1001),
    (0x0010, 0x2160), (0x0010, 0x21B0), (0x0018, 0x1030), (0x0008, 0x1070),
    (0x0008, 0x009C),
]

SENSITIVE_KEYWORDS = ['name', 'id', 'dob', 'birth', 'mrn', 'patient', 'hospital']
DATE_PATTERN = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')
ID_PATTERN = re.compile(r'\b\d{5,}\b')  # IDs: 5+ digits

reader = easyocr.Reader(['en'])

def is_sensitive_text(text):
    t = text.lower()
    return (
        any(k in t for k in SENSITIVE_KEYWORDS) or
        DATE_PATTERN.search(t) or
        ID_PATTERN.search(t)
    )

def redact_burned_in_text(ds):
    pixel_array = ds.pixel_array
    img = pixel_array.astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    results = reader.readtext(img)
    redacted_count = 0

    for (bbox, text, conf) in results:
        if is_sensitive_text(text):
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)
            redacted_count += 1
            print(f"🕶️ Redacted burned-in text: {text}")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ds.PixelData = gray_img.tobytes()
    ds.Rows, ds.Columns = gray_img.shape
    return redacted_count

def anonymize_dicom(input_path, output_path):
    ds = pydicom.dcmread(input_path)
    print(f"\n📄 Processing: {input_path}")

    # Step 1: Tag-level anonymization
    for tag in SENSITIVE_TAGS:
        if tag in ds:
            print(f"❌ Removing tag {ds[tag].name}: {ds[tag].value}")
            del ds[tag]

    ds.remove_private_tags()

    if 'StudyInstanceUID' in ds:
        ds.StudyInstanceUID = generate_uid()
    if 'SeriesInstanceUID' in ds:
        ds.SeriesInstanceUID = generate_uid()
    if 'SOPInstanceUID' in ds:
        ds.SOPInstanceUID = generate_uid()

    # Step 2: Burned-in text removal
    if hasattr(ds, 'PixelData'):
        count = redact_burned_in_text(ds)
        if count:
            print(f"🔏 Redacted {count} burned-in texts.")

    if 'BurnedInAnnotation' in ds:
        ds.BurnedInAnnotation = 'NO'

    ds.save_as(output_path)
    print(f"✅ Anonymized file saved as: {output_path}")