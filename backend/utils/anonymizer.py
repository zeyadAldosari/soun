import pydicom
import cv2
import numpy as np
import easyocr
import cv2
import numpy as np
import easyocr
from pydicom.uid import generate_uid
import re
import os
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
import logging
from pathlib import Path
import copy

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import pixel data handlers - don't fail if some are missing
try:
    import pydicom.pixel_data_handlers.pylibjpeg_handler as pylibjpeg_handler
    pydicom.config.pixel_data_handlers.append(pylibjpeg_handler)
    logger.info("PyLibJPEG pixel data handler loaded - handles JPEG, JPEG-LS, JPEG2000")
except ImportError:
    logger.warning("PyLibJPEG pixel data handler not available - compressed files may not load correctly")
    logger.warning("Install with: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg pylibjpeg-rle")

try:
    import pydicom.pixel_data_handlers.pillow_handler as pillow_handler
    pydicom.config.pixel_data_handlers.append(pillow_handler)
    logger.info("Pillow pixel data handler loaded")
except ImportError:
    logger.warning("Pillow pixel data handler not available")

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
    (0x0010, 0x0101),  # Patient's Primary Language Code Sequence
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

# Extended keywords for OCR text detection
SENSITIVE_KEYWORDS = [
    'name', 'id', 'dob', 'birth', 'mrn', 'patient', 'hospital', 'philips', 'ge', 'siemens',
    'healthcare', 'clinic', 'dr.', 'dr ', 'doctor', 'physician', 'accession', 'medical',
    'record', 'ssn', 'social', 'security', 'address', 'phone', 'study', 'exam', 'sex',
    'age', 'male', 'female', 'insurance', 'provider', 'account', 'admission'
]

# Regex patterns for sensitive information
DATE_PATTERN = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')  # Dates like MM/DD/YYYY
ID_PATTERN = re.compile(r'\b\d{5,}\b')  # IDs: 5+ digits
SSN_PATTERN = re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b')  # SSN: xxx-xx-xxxx
PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')  # Phone: xxx-xxx-xxxx
TIME_PATTERN = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b')  # Times

class DicomAnonymizer:
    def __init__(self, ocr_languages=['en'], ocr_gpu=False, confidence_threshold=0.5):
        """
        Initialize the DICOM anonymizer.
        
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
        """
        Check if text contains sensitive information.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if sensitive, False otherwise
        """
        if not text or len(text) < 2:
            return False
            
        t = text.lower()
        return (
            any(k in t for k in SENSITIVE_KEYWORDS) or
            DATE_PATTERN.search(t) or
            ID_PATTERN.search(t) or
            SSN_PATTERN.search(t) or
            PHONE_PATTERN.search(t) or
            TIME_PATTERN.search(t)
        )
    
    def preprocess_image(self, img, enhance_contrast=True):
        """
        Preprocess image for OCR.
        
        Args:
            img: Image to preprocess
            enhance_contrast: Whether to enhance contrast
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Enhance contrast if requested
        if enhance_contrast:
            gray = cv2.equalizeHist(gray)
        
        # Apply additional preprocessing for better OCR
        # Bilateral filter preserves edges while reducing noise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        return gray
    
    def extract_pixel_array(self, ds):
        """
        Extract and normalize pixel array from DICOM dataset.
        
        Args:
            ds: DICOM dataset
            
        Returns:
            Normalized image array suitable for processing
        """
        try:
            # Get original pixel data
            pixel_array = ds.pixel_array
            
            # Handle different photometric interpretations
            photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
            
            # Apply modality LUT if present (e.g., for CT Hounsfield units)
            if hasattr(ds, 'RescaleSlope') or hasattr(ds, 'RescaleIntercept'):
                pixel_array = apply_modality_lut(pixel_array, ds)
            
            # Apply VOI LUT for better visualization if present
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                pixel_array = apply_voi_lut(pixel_array, ds)
            
            # Normalize to 0-255 range for display and OCR
            if pixel_array.max() > 255 or pixel_array.min() < 0:
                pixel_min = pixel_array.min()
                pixel_max = pixel_array.max()
                if pixel_max > pixel_min:
                    pixel_array = 255 * (pixel_array - pixel_min) / (pixel_max - pixel_min)
            
            # Convert to uint8 for OpenCV operations
            pixel_array = pixel_array.astype(np.uint8)
            
            # Convert to BGR for OpenCV if grayscale
            if len(pixel_array.shape) == 2:
                pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
            
            return pixel_array
            
        except Exception as e:
            logger.error(f"Error extracting pixel array: {e}")
            raise
    
    def redact_burned_in_text(self, ds):
        """
        Detect and redact burned-in text from DICOM images.
        
        Args:
            ds: DICOM dataset
            
        Returns:
            int: Number of redacted text regions
        """
        if not hasattr(ds, 'PixelData'):
            logger.warning("No pixel data found in DICOM file")
            return 0
        
        try:
            # Store original transfer syntax
            original_transfer_syntax = ds.file_meta.TransferSyntaxUID
            is_compressed = original_transfer_syntax != '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
            
            if is_compressed:
                logger.info(f"Processing compressed DICOM with transfer syntax: {original_transfer_syntax}")
            
            # Extract normalized pixel array for processing
            img = self.extract_pixel_array(ds)
            
            # Check if this is a multi-frame image
            is_multiframe = False
            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                is_multiframe = True
                logger.info(f"Multi-frame image detected with {ds.NumberOfFrames} frames")
                logger.warning("Multi-frame support is limited - processing only first frame")
                
            # Process the image for OCR
            preprocessed = self.preprocess_image(img)
            results = self.reader.readtext(preprocessed)
            redacted_count = 0
            
            # Keep track of all redacted areas
            redacted_regions = []
            
            for (bbox, text, conf) in results:
                # Skip low confidence detections
                if conf < self.confidence_threshold:
                    continue
                    
                logger.debug(f"OCR detected: '{text}' with confidence {conf:.2f}")
                
                if self.is_sensitive_text(text):
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    
                    # Add some padding to ensure complete redaction
                    padding = 5
                    top_left = (max(0, top_left[0] - padding), max(0, top_left[1] - padding))
                    bottom_right = (min(img.shape[1], bottom_right[0] + padding), 
                                   min(img.shape[0], bottom_right[1] + padding))
                    
                    # Store redacted region
                    redacted_regions.append((top_left, bottom_right))
                    
                    # Redact with black rectangle
                    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)
                    redacted_count += 1
                    logger.info(f"Redacted text: '{text}'")
            
            if redacted_count > 0:
                # Convert back to appropriate format for DICOM
                if len(ds.pixel_array.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # For compressed transfer syntaxes, we need to handle this differently
                # Always convert to uncompressed transfer syntax after modifying pixel data
                ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
                
                # Update pixel data
                if len(img.shape) == 2:
                    # For grayscale images
                    ds.Rows = img.shape[0]
                    ds.Columns = img.shape[1]
                    ds.SamplesPerPixel = 1
                    ds.BitsAllocated = 8
                    ds.BitsStored = 8
                    ds.HighBit = 7
                    ds.PixelRepresentation = 0
                    ds.PixelData = img.tobytes()
                else:
                    # For color images
                    ds.Rows = img.shape[0]
                    ds.Columns = img.shape[1]
                    ds.SamplesPerPixel = 3
                    ds.BitsAllocated = 8
                    ds.BitsStored = 8
                    ds.HighBit = 7
                    ds.PixelRepresentation = 0
                    ds.PhotometricInterpretation = "RGB"
                    ds.PlanarConfiguration = 0  # Color-by-pixel
                    ds.PixelData = img.tobytes()
                
                # Update metadata if needed
                if hasattr(ds, 'BurnedInAnnotation'):
                    ds.BurnedInAnnotation = 'NO'
                
                # Update our statistics
                self.stats['redacted_text_regions'] += redacted_count
            
            return redacted_count
            
        except Exception as e:
            logger.error(f"Error redacting burned-in text: {e}")
            self.stats['errors'] += 1
            return 0
    
    def anonymize_dataset(self, ds, keep_uids=False):
        """
        Anonymize a DICOM dataset.
        
        Args:
            ds: DICOM dataset to anonymize
            keep_uids: Whether to keep original UIDs
            
        Returns:
            Anonymized DICOM dataset
        """
        # Clone dataset to avoid modifying the original
        ds_anon = ds.copy()
        
        # Step 1: Remove sensitive tags
        tags_removed = 0
        for tag in SENSITIVE_TAGS:
            if tag in ds_anon:
                logger.debug(f"Removing tag {ds_anon[tag].name}: {ds_anon[tag].value}")
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
    
    def anonymize_file(self, input_path, output_path, redact_overlays=True, keep_uids=False,
                       force_uncompressed=True):
        """
        Anonymize a DICOM file and save to output path.
        
        Args:
            input_path: Path to input DICOM file
            output_path: Path to output anonymized DICOM file
            redact_overlays: Whether to attempt redaction of burned-in text
            keep_uids: Whether to keep original UIDs
            force_uncompressed: Convert to uncompressed transfer syntax
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Processing: {input_path}")
            
            # Read DICOM file - force_read=True to bypass errors
            try:
                ds = pydicom.dcmread(input_path, force=True)
            except Exception as e:
                logger.error(f"Error reading DICOM file: {e}")
                logger.info("Trying to read file with different pixel data handlers...")
                
                # If we fail, try with different force parameters
                ds = pydicom.dcmread(input_path, force=True)
            
            # Check for compression
            original_ts = ds.file_meta.TransferSyntaxUID
            is_compressed = original_ts != '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
            
            if is_compressed:
                logger.info(f"File uses compressed transfer syntax: {original_ts}")
                logger.info("Recommended dependencies for this file:")
                
                if "1.2.840.10008.1.2.4.50" in original_ts or "1.2.840.10008.1.2.4.51" in original_ts:
                    # JPEG Baseline/Extended
                    logger.info("JPEG compression detected - install: pip install pylibjpeg pylibjpeg-libjpeg")
                elif "1.2.840.10008.1.2.4.80" in original_ts or "1.2.840.10008.1.2.4.81" in original_ts:
                    # JPEG-LS
                    logger.info("JPEG-LS compression detected - install: pip install pylibjpeg pylibjpeg-libjpeg")
                elif "1.2.840.10008.1.2.4.90" in original_ts or "1.2.840.10008.1.2.4.91" in original_ts:
                    # JPEG 2000
                    logger.info("JPEG 2000 compression detected - install: pip install pylibjpeg pylibjpeg-openjpeg")
                elif "1.2.840.10008.1.2.5" in original_ts:
                    # RLE
                    logger.info("RLE compression detected - install: pip install pylibjpeg pylibjpeg-rle")
            
            # Anonymize dataset
            ds_anon = self.anonymize_dataset(ds, keep_uids)
            
            # Redact burned-in text if requested
            redacted_count = 0
            if redact_overlays and hasattr(ds_anon, 'PixelData'):
                try:
                    redacted_count = self.redact_burned_in_text(ds_anon)
                    if redacted_count > 0:
                        logger.info(f"Redacted {redacted_count} text regions")
                    else:
                        logger.info("No sensitive burned-in text detected")
                except Exception as e:
                    logger.error(f"Error during text redaction: {e}")
                    self.stats['errors'] += 1
            
            # Force uncompressed transfer syntax if requested
            if force_uncompressed and is_compressed and not (redacted_count > 0):  # If redacted, we've already changed the TS
                logger.info("Converting to uncompressed transfer syntax")
                ds_anon.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
            
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save anonymized file
            ds_anon.save_as(output_path)
            logger.info(f"Anonymized file saved as: {output_path}")
            
            # Update statistics
            self.stats['processed_files'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error anonymizing file {input_path}: {e}")
            self.stats['errors'] += 1
            self.stats['skipped_files'] += 1
            return False
    
    def anonymize_directory(self, input_dir, output_dir, recursive=True, file_pattern='*.dcm',
                           redact_overlays=True, keep_uids=False, force_uncompressed=True):
        """
        Anonymize all DICOM files in a directory.
        
        Args:
            input_dir: Input directory containing DICOM files
            output_dir: Output directory for anonymized files
            recursive: Whether to process subdirectories
            file_pattern: File pattern to match DICOM files
            redact_overlays: Whether to attempt redaction of burned-in text
            keep_uids: Whether to keep original UIDs
            force_uncompressed: Convert to uncompressed transfer syntax
            
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
                keep_uids=keep_uids,
                force_uncompressed=force_uncompressed
            )
        
        # Log final statistics
        logger.info(f"Anonymization complete. Statistics:")
        logger.info(f"  Processed files: {self.stats['processed_files']}")
        logger.info(f"  Skipped files: {self.stats['skipped_files']}")
        logger.info(f"  Removed tags: {self.stats['removed_tags']}")
        logger.info(f"  Redacted text regions: {self.stats['redacted_text_regions']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        
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
    parser.add_argument('--keep-compressed', action='store_true', 
                        help='Keep original compression (may fail with some files)')
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
            force_uncompressed=not args.keep_compressed
        )
    else:
        # Process single file
        anonymizer.anonymize_file(
            args.input, 
            args.output, 
            redact_overlays=not args.no_redact,
            keep_uids=args.keep_uids,
            force_uncompressed=not args.keep_compressed
        )