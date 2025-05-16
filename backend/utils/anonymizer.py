import pydicom
import cv2
import numpy as np
import easyocr
from pydicom.uid import generate_uid
import re
import os
import logging
from pathlib import Path
import copy
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import pixel data handlers - don't fail if some are missing
try:
    import pydicom.pixel_data_handlers.numpy_handler as numpy_handler
    pydicom.config.pixel_data_handlers.append(numpy_handler)
    logger.info("NumPy pixel data handler loaded")
except ImportError:
    logger.warning("NumPy pixel data handler not available")

# Extended list of sensitive DICOM tags
SENSITIVE_TAGS = [
    # Patient info
    (0x0010, 0x0010),  # Patient's Name
    (0x0010, 0x0020),  # Patient ID
    (0x0010, 0x0030),  # Patient's Birth Date
    (0x0010, 0x0040),  # Patient's Sex
    (0x0010, 0x1010),  # Patient's Age
    (0x0010, 0x1040),  # Patient's Address
    (0x0010, 0x0032),  # Patient's Birth Time
    (0x0010, 0x0050),  # Patient's Insurance Plan Code Sequence
    (0x0010, 0x1000),  # Other Patient IDs
    (0x0010, 0x1001),  # Other Patient Names
    (0x0010, 0x1005),  # Patient's Birth Name
    (0x0010, 0x1060),  # Patient's Mother's Birth Name
    (0x0010, 0x1080),  # Military Rank
    (0x0010, 0x1090),  # Medical Record Locator
    (0x0010, 0x2000),  # Medical Alerts
    (0x0010, 0x2110),  # Allergies
    (0x0010, 0x2150),  # Country of Residence
    (0x0010, 0x2152),  # Region of Residence
    (0x0010, 0x2154),  # Patient's Telephone Numbers
    (0x0010, 0x2160),  # Ethnic Group
    (0x0010, 0x2180),  # Occupation
    (0x0010, 0x21B0),  # Additional Patient History
    (0x0010, 0x21C0),  # Pregnancy Status
    (0x0010, 0x21D0),  # Last Menstrual Date
    (0x0010, 0x4000),  # Patient Comments
    
    # Institution info
    (0x0008, 0x0080),  # Institution Name
    (0x0008, 0x0081),  # Institution Address
    (0x0008, 0x0082),  # Institution Code Sequence
    (0x0008, 0x0090),  # Referring Physician's Name
    (0x0008, 0x0092),  # Referring Physician's Address
    (0x0008, 0x0094),  # Referring Physician's Telephone Numbers
    (0x0008, 0x0096),  # Referring Physician ID Sequence
    (0x0008, 0x1010),  # Station Name
    (0x0008, 0x1030),  # Study Description
    (0x0008, 0x1048),  # Physician(s) of Record
    (0x0008, 0x1049),  # Physician(s) of Record ID Sequence
    (0x0008, 0x1050),  # Performing Physician's Name
    (0x0008, 0x1060),  # Name of Physician(s) Reading Study
    (0x0008, 0x1070),  # Operators' Name
    (0x0008, 0x1080),  # Admitting Diagnoses Description
    (0x0008, 0x1155),  # Referenced SOP Instance UID
    (0x0008, 0x2111),  # Derivation Description
    
    # Study info
    (0x0008, 0x0050),  # Accession Number
    (0x0020, 0x000D),  # Study Instance UID
    (0x0020, 0x0010),  # Study ID
    (0x0020, 0x0052),  # Frame of Reference UID
    (0x0020, 0x0200),  # Synchronization Frame of Reference UID
    
    # Series info
    (0x0008, 0x009C),  # Consulting Physician's Name
    (0x0018, 0x1030),  # Protocol Name
    (0x0020, 0x000E),  # Series Instance UID
    (0x0020, 0x0011),  # Series Number
    
    # Equipment info
    (0x0018, 0x1000),  # Device Serial Number
    (0x0018, 0x1020),  # Software Versions
]

# Keywords for OCR text detection - simplified but comprehensive
SENSITIVE_KEYWORDS = [
    'name', 'id', 'dob', 'birth', 'mrn', 'patient', 'hospital', 'philips', 'ge', 'siemens',
    'healthcare', 'clinic', 'dr.', 'dr ', 'doctor', 'physician', 'medical', 'record',
    'address', 'phone', 'study', 'exam', 'sex', 'age', 'male', 'female', 'insurance'
]

# Regex patterns for sensitive information
DATE_PATTERN = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')  # Dates like MM/DD/YYYY
ID_PATTERN = re.compile(r'\b\d{5,}\b')  # IDs: 5+ digits
SSN_PATTERN = re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b')  # SSN: xxx-xx-xxxx
PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')  # Phone: xxx-xxx-xxxx

class DicomAnonymizer:
    def __init__(self, ocr_languages=['en'], ocr_gpu=False, confidence_threshold=0.4):
        """
        Initialize the DICOM anonymizer with simplified approach.
        
        Args:
            ocr_languages: List of languages for OCR
            ocr_gpu: Whether to use GPU for OCR
            confidence_threshold: Minimum confidence for OCR detections
        """
        self.reader = easyocr.Reader(ocr_languages, gpu=ocr_gpu)
        self.confidence_threshold = confidence_threshold
        self.stats = {
            'processed_files': 0,
            'skipped_files': 0,
            'removed_tags': 0,
            'redacted_text_regions': 0,
            'errors': 0
        }
    
    def is_sensitive_text(self, text):
        """Check if text contains sensitive information"""
        if not text or len(text) < 2:
            return False
            
        t = text.lower()
        # Check for hospital names, patient names, etc.
        name_pattern = re.compile(r'(?i)(bernard|sophie|patient|dr\.|dr\s|hopital|hospital|clinic|philips|healthcare)', re.IGNORECASE)
        
        # Image-specific pattern for ultrasound headers (e.g., dates, IDs in formats like 11-05-25-142825)
        ultrasound_id_pattern = re.compile(r'\d{2}-\d{2}-\d{2}-\d{6}')
        
        return (
            any(k in t for k in SENSITIVE_KEYWORDS) or
            name_pattern.search(t) or
            DATE_PATTERN.search(t) or
            ID_PATTERN.search(t) or
            SSN_PATTERN.search(t) or
            PHONE_PATTERN.search(t) or
            ultrasound_id_pattern.search(text)  # Use original text to maintain case
        )
    
    def preprocess_image(self, img):
        """
        Improved preprocessing for OCR with better handling of color backgrounds
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply adaptive histogram equalization for better text detection in varying backgrounds
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_equalized = clahe.apply(gray)
        
        # Create a second version with more aggressive equalization for dark backgrounds
        clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray_strong = clahe_strong.apply(gray)
        
        # Apply mild bilateral filter to reduce noise while preserving edges
        gray_filtered = cv2.bilateralFilter(gray_equalized, 5, 10, 10)
        
        # Return both versions for processing
        return gray_filtered, gray_strong
    
    def redact_burned_in_text(self, ds):
        """
        Simplified approach to detect and redact burned-in text
        """
        if not hasattr(ds, 'PixelData'):
            logger.warning("⚠️ No pixel data found in DICOM file")
            return 0
        
        try:
            # Extract pixel array
            original_pixel_array = ds.pixel_array.copy()
            
            # Convert to 8-bit for processing if needed
            if original_pixel_array.dtype != np.uint8:
                # Scale to 0-255 range
                pixel_min = original_pixel_array.min()
                pixel_max = original_pixel_array.max()
                if pixel_max > pixel_min:
                    img = ((original_pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(original_pixel_array, dtype=np.uint8)
            else:
                img = original_pixel_array.copy()
            
            # Convert to BGR for processing if grayscale
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                # Assume it's already in a format we can use
                img_bgr = img
            
            # Process the header area separately for higher sensitivity
            header_height = int(img_bgr.shape[0] * 0.15)  # Top 15% of image
            header_img = img_bgr[0:header_height, :]
            
            # Process image for OCR - improved approach for different background colors
            preprocessed_img, preprocessed_strong = self.preprocess_image(img_bgr)
            
            # Process the header area separately for better detection
            header_height = int(img_bgr.shape[0] * 0.15)  # Top 15% of image
            
            # Run OCR on both preprocessed versions
            results = self.reader.readtext(preprocessed_img)
            
            # Additional OCR on strongly processed version for dark backgrounds
            results_strong = self.reader.readtext(preprocessed_strong)
            
            # Combine results, removing duplicates
            for result in results_strong:
                if result not in results:
                    results.append(result)
            
            # Check for header text with lower threshold
            header_text_detected = False
            for (bbox, text, conf) in results:
                if bbox[0][1] < header_height and conf >= max(0.2, self.confidence_threshold - 0.2):
                    if self.is_sensitive_text(text):
                        header_text_detected = True
                        break
            
            # Track redacted regions
            redacted_count = 0
            
            # Redact header if needed
            if header_text_detected:
                logger.info("🔒 Redacting header area containing possible patient information")
                
                # Detect header background color to choose appropriate redaction color
                header_region = img_bgr[0:header_height, :]
                avg_color = np.mean(header_region, axis=(0, 1))
                
                # If the header is dark (like blue/black), use white for redaction
                # Otherwise use black for redaction
                is_dark_header = np.mean(avg_color) < 128
                
                if is_dark_header:
                    logger.info("Detected dark header, using white redaction")
                    redaction_color = (255, 255, 255)  # White
                else:
                    redaction_color = (0, 0, 0)  # Black
                
                # Apply redaction with detected color
                cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], header_height), redaction_color, -1)
                redacted_count += 1
            
            # Process all detected text
            for (bbox, text, conf) in results:
                logger.info(f"OCR detected: '{text}' with confidence {conf:.2f}")
                
                if conf < self.confidence_threshold:
                    continue
                
                if self.is_sensitive_text(text):
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    
                    # Add padding to ensure complete redaction
                    padding = 5
                    top_left = (max(0, top_left[0] - padding), max(0, top_left[1] - padding))
                    bottom_right = (min(img_bgr.shape[1], bottom_right[0] + padding), 
                                   min(img_bgr.shape[0], bottom_right[1] + padding))
                    
                    # Determine if this region has a dark background
                    region = img_bgr[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    if region.size > 0:  # Ensure region is valid
                        avg_color = np.mean(region, axis=(0, 1)) if region.size > 0 else np.array([0, 0, 0])
                        is_dark_region = np.mean(avg_color) < 128
                        
                        # Choose redaction color based on background
                        if is_dark_region:
                            redaction_color = (255, 255, 255)  # White for dark backgrounds
                        else:
                            redaction_color = (0, 0, 0)  # Black for light backgrounds
                    else:
                        redaction_color = (0, 0, 0)  # Default to black
                    
                    # Apply redaction with appropriate color
                    cv2.rectangle(img_bgr, top_left, bottom_right, redaction_color, -1)
                    redacted_count += 1
                    logger.info(f"🔒 Redacted text: '{text}' with confidence {conf:.2f}")
            
            if redacted_count > 0:
                # Convert back to grayscale if needed
                if len(original_pixel_array.shape) == 2:
                    img_processed = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                else:
                    img_processed = img_bgr
                
                # Update DICOM file with redacted image, always using uncompressed transfer syntax
                ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
                
                # Preserve original pixel data characteristics
                if original_pixel_array.dtype != np.uint8:
                    # Convert back to original data range and type
                    if img_processed.dtype == np.uint8 and original_pixel_array.dtype != np.uint8:
                        # Scale back to original range
                        img_float = img_processed.astype(np.float32) / 255.0
                        img_scaled = img_float * (pixel_max - pixel_min) + pixel_min
                        img_final = img_scaled.astype(original_pixel_array.dtype)
                    else:
                        img_final = img_processed
                else:
                    img_final = img_processed
                
                # Update pixel data
                ds.PixelData = img_final.tobytes()
                
                # Update metadata to ensure consistency
                ds.Rows = img_final.shape[0]
                ds.Columns = img_final.shape[1]
                
                # Mark burned-in annotations as removed
                if hasattr(ds, 'BurnedInAnnotation'):
                    ds.BurnedInAnnotation = 'NO'
                
                # Update our statistics
                self.stats['redacted_text_regions'] += redacted_count
            
            return redacted_count
            
        except Exception as e:
            logger.error(f"❌ Error redacting burned-in text: {e}")
            self.stats['errors'] += 1
            return 0
    
    def anonymize_dataset(self, ds, keep_uids=False):
        """
        Anonymize a DICOM dataset by removing sensitive tags
        """
        # Clone dataset to avoid modifying the original
        ds_anon = ds.copy()
        
        # Step 1: Remove sensitive tags
        tags_removed = 0
        for tag in SENSITIVE_TAGS:
            if tag in ds_anon:
                tag_name = ds_anon[tag].name
                tag_value = str(ds_anon[tag].value)
                # Mask the value for privacy in logs if it's not empty
                if tag_value.strip():
                    # Keep first character and mask the rest for patient info
                    if "Patient" in tag_name and len(tag_value) > 1:
                        if "Date" not in tag_name:  # Don't mask dates
                            masked_value = tag_value[0] + "*" * (len(tag_value) - 1)
                        else:
                            masked_value = tag_value
                    else:
                        masked_value = tag_value
                else:
                    masked_value = ""
                
                logger.info(f"❌ Removing tag {tag_name}: {masked_value}")
                del ds_anon[tag]
                tags_removed += 1
        
        # Step 2: Remove private tags
        ds_anon.remove_private_tags()
        
        # Step 3: Update UIDs if requested
        if not keep_uids:
            # Generate new UIDs
            if 'StudyInstanceUID' in ds_anon:
                ds_anon.StudyInstanceUID = generate_uid()
            if 'SeriesInstanceUID' in ds_anon:
                ds_anon.SeriesInstanceUID = generate_uid()
            if 'SOPInstanceUID' in ds_anon:
                ds_anon.SOPInstanceUID = generate_uid()
        
        # Update statistics
        self.stats['removed_tags'] += tags_removed
        
        return ds_anon
    
    def anonymize_file(self, input_path, output_path, redact_overlays=True, keep_uids=False):
        """
        Anonymize a DICOM file and save to output path - simplified approach
        
        Args:
            input_path: Path to input DICOM file
            output_path: Path to output anonymized DICOM file
            redact_overlays: Whether to attempt redaction of burned-in text
            keep_uids: Whether to keep original UIDs
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"📄 Processing: {input_path}")
            logger.info(f"📤 Output will be saved to: {output_path}")
            
            # Read DICOM file - force=True to bypass some errors
            try:
                ds = pydicom.dcmread(input_path, force=True)
            except Exception as e:
                logger.error(f"❌ Error reading DICOM file: {e}")
                self.stats['errors'] += 1
                self.stats['skipped_files'] += 1
                return False
            
            # Check if there's pixel data
            has_pixel_data = hasattr(ds, 'PixelData')
            if not has_pixel_data:
                logger.warning("⚠️ DICOM file has no pixel data - OCR and redaction will be skipped")
            
            # Create a copy for anonymization
            ds_copy = copy.deepcopy(ds)
            
            # First anonymize dataset metadata
            ds_anon = self.anonymize_dataset(ds_copy, keep_uids)
            
            # Remember original transfer syntax
            original_syntax = None
            if hasattr(ds_anon, 'file_meta') and hasattr(ds_anon.file_meta, 'TransferSyntaxUID'):
                original_syntax = ds_anon.file_meta.TransferSyntaxUID
            
            # Redact burned-in text if requested
            redacted_count = 0
            if redact_overlays and has_pixel_data:
                try:
                    logger.info("🔍 Starting OCR detection for burned-in text...")
                    redacted_count = self.redact_burned_in_text(ds_anon)
                    
                    if redacted_count > 0:
                        logger.info(f"Redacted {redacted_count} text regions ✅")
                        # Redaction already sets to uncompressed transfer syntax 1.2.840.10008.1.2.1
                    else:
                        logger.info("No sensitive burned-in text detected ✅")
                        # If we didn't redact anything, restore original transfer syntax
                        if original_syntax:
                            ds_anon.file_meta.TransferSyntaxUID = original_syntax
                except Exception as e:
                    logger.error(f"❌ Error during text redaction: {e}")
                    self.stats['errors'] += 1
            
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Add a "modified" tag to track processing
            current_time = datetime.now().time()
            formatted_time = current_time.strftime('%H%M%S.%f')
            ds_anon.add_new([0x0008, 0x0031], 'TM', formatted_time)
            
            try:
                # Save anonymized file
                ds_anon.save_as(output_path)
                logger.info(f"✅ Anonymized file saved as: {output_path}")
                
                # Update statistics
                self.stats['processed_files'] += 1
                return True
            except Exception as e:
                logger.error(f"❌ Error saving file: {e}")
                
                # Try one more time with uncompressed transfer syntax
                try:
                    logger.warning("⚠️ Trying to save with uncompressed format...")
                    ds_anon.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
                    ds_anon.save_as(output_path)
                    logger.info(f"✅ Saved file in uncompressed format: {output_path}")
                    return True
                except Exception as e2:
                    logger.error(f"❌ Final error saving file: {e2}")
                    self.stats['errors'] += 1
                    self.stats['skipped_files'] += 1
                    return False
        
        except Exception as e:
            logger.error(f"❌ Error anonymizing file {input_path}: {e}")
            self.stats['errors'] += 1
            self.stats['skipped_files'] += 1
            return False
    
    def anonymize_directory(self, input_dir, output_dir, recursive=True, file_pattern='*.dcm',
                           redact_overlays=True, keep_uids=False):
        """
        Anonymize all DICOM files in a directory
        
        Args:
            input_dir: Input directory containing DICOM files
            output_dir: Output directory for anonymized files
            recursive: Whether to process subdirectories
            file_pattern: File pattern to match DICOM files
            redact_overlays: Whether to attempt redaction of burned-in text
            keep_uids: Whether to keep original UIDs
            
        Returns:
            dict: Statistics about the processing
        """
        # Reset statistics
        self.stats = {
            'processed_files': 0,
            'skipped_files': 0,
            'removed_tags': 0,
            'redacted_text_regions': 0,
            'errors': 0
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process files
        input_path = Path(input_dir)
        
        # Get all matching files
        if recursive:
            files = list(input_path.glob(f"**/{file_pattern}"))
        else:
            files = list(input_path.glob(file_pattern))
        
        logger.info(f"Found {len(files)} DICOM files to process")
        
        # Process each file
        for file_path in files:
            # Determine output path
            rel_path = file_path.relative_to(input_path)
            output_path = Path(output_dir) / rel_path
            
            # Make sure output directory exists
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Anonymize file
            self.anonymize_file(
                str(file_path), 
                str(output_path),
                redact_overlays=redact_overlays,
                keep_uids=keep_uids
            )
        
        # Log final statistics
        logger.info(f"📊 Anonymization complete. Statistics:")
        logger.info(f"  ✅ Processed files: {self.stats['processed_files']}")
        logger.info(f"  ⚠️ Skipped files: {self.stats['skipped_files']}")
        logger.info(f"  🔒 Removed tags: {self.stats['removed_tags']}")
        logger.info(f"  🔒 Redacted text regions: {self.stats['redacted_text_regions']}")
        logger.info(f"  ❌ Errors: {self.stats['errors']}")
        
        return self.stats

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DICOM Anonymizer')
    parser.add_argument('--input', required=True, help='Input DICOM file or directory')
    parser.add_argument('--output', required=True, help='Output DICOM file or directory')
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--pattern', default='*.dcm', help='File pattern for DICOM files')
    parser.add_argument('--keep-uids', action='store_true', help='Keep original UIDs')
    parser.add_argument('--no-redact', action='store_true', help='Skip redaction of burned-in text')
    parser.add_argument('--log-level', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create anonymizer
    anonymizer = DicomAnonymizer()
    
    # Process input
    if os.path.isdir(args.input):
        # Process directory
        anonymizer.anonymize_directory(
            args.input, 
            args.output, 
            recursive=args.recursive, 
            file_pattern=args.pattern,
            redact_overlays=not args.no_redact,
            keep_uids=args.keep_uids,
      
        )
    else:
        # Process single file
        anonymizer.anonymize_file(
            args.input, 
            args.output, 
            redact_overlays=not args.no_redact,
            keep_uids=args.keep_uids
        )