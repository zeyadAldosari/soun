import pydicom
import cv2
import numpy as np
import easyocr
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID
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

# Try to import pixel data handlers
try:
    import pydicom.pixel_data_handlers.numpy_handler as numpy_handler
    pydicom.config.pixel_data_handlers.append(numpy_handler)
    logger.info("NumPy pixel data handler loaded")
except ImportError:
    logger.warning("NumPy pixel data handler not available")

# Extended list of sensitive DICOM tags
SENSITIVE_TAGS = [
    (0x0010, 0x0010), (0x0010, 0x0020), (0x0010, 0x0030), (0x0010, 0x0040),
    (0x0010, 0x1010), (0x0010, 0x1040), (0x0010, 0x0032), (0x0010, 0x0050),
    (0x0010, 0x1000), (0x0010, 0x1001), (0x0010, 0x1005), (0x0010, 0x1060),
    (0x0010, 0x1080), (0x0010, 0x1090), (0x0010, 0x2000), (0x0010, 0x2110),
    (0x0010, 0x2150), (0x0010, 0x2152), (0x0010, 0x2154), (0x0010, 0x2160),
    (0x0010, 0x2180), (0x0010, 0x21B0), (0x0010, 0x21C0), (0x0010, 0x21D0),
    (0x0010, 0x4000), (0x0008, 0x0080), (0x0008, 0x0081), (0x0008, 0x0082),
    (0x0008, 0x0090), (0x0008, 0x0092), (0x0008, 0x0094), (0x0008, 0x0096),
    (0x0008, 0x1010), (0x0008, 0x1030), (0x0008, 0x1048), (0x0008, 0x1049),
    (0x0008, 0x1050), (0x0008, 0x1060), (0x0008, 0x1070), (0x0008, 0x1080),
    (0x0008, 0x1155), (0x0008, 0x2111), (0x0008, 0x0050), (0x0020, 0x000D),
    (0x0020, 0x0010), (0x0020, 0x0052), (0x0020, 0x0200), (0x0008, 0x009C),
    (0x0018, 0x1030), (0x0020, 0x000E), (0x0020, 0x0011), (0x0018, 0x1000),
    (0x0018, 0x1020),
]


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Refined sensitive keywords - more specific to reduce false positives
SENSITIVE_KEYWORDS_REFINED = [
    # Patient identifiers
    r'\bpatient\s*(?:name|id)\b', r'\bmrn\b', r'\bmedical\s*record\s*number\b', 
    r'\baccession\s*(?:number|#)\b', r'\buid\b', r'\bfacility\s*id\b',
    r'\bencounter\s*(?:number|id)\b', r'\bvisit\s*(?:number|id)\b',
    
    # Personal identifiers
    r'\bssn\b', r'\bsocial\s*security\b', r'\bdob\b', r'\bdate\s*of\s*birth\b',
    r'\bdriver\'?s?\s*license\b', r'\bpassport\b', r'\bfax\b',
    
    # Contact information
    r'\bemail\b', r'\be-mail\b', r'\baddress\b', r'\bzip\s*code\b', r'\bcity\b', r'\bstate\b',
    r'\bcounty\b', r'\bcountry\b', r'\bpostal\s*code\b',
    
    # Healthcare facilities
    r'\bhospital\b', r'\bclinic\b', r'\bcenter\b', r'\binstitution\b', r'\bfacility\b',
    r'\blaboratory\b', r'\blab\b', r'\bpharmacy\b', r'\bpractice\b', r'\boffice\b',
    
    # Healthcare providers
    r'\bdr\.?\s*(?:[a-z]+\s+)?[a-z]+\b', r'\bdoctor\s+[a-z]+\b', r'\bphysician\b',
    r'\bnurse\b', r'\bRN\b', r'\bLPN\b', r'\bNP\b', r'\bPA\b', r'\btherapist\b',
    
    # Insurance
    r'\binsurance\b', r'\bpolicy\s*(?:number|#)\b', r'\bgroup\s*(?:number|#)\b',
    r'\bsubscriber\b', r'\bbeneficiary\b', r'\bcoverage\b',
    
    # Explicit name indicators
    r'\blast\s*name\b', r'\bfirst\s*name\b', r'\bmiddle\s*name\b', r'\bfull\s*name\b',
    r'\bname\s*of\s*patient\b', r'\bpatient\'?s?\s*name\b', r'\bguardian\b', r'\bnext\s*of\s*kin\b'
]

# Date patterns - comprehensive
DATE_PATTERN = re.compile(
    r'\b('
    # Standard formats: MM/DD/YYYY, YYYY-MM-DD, etc.
    r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|'
    r'\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|'
    # Month name formats: Jan 1, 2023; 1st Jan 2023
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-.\s]+\d{1,2}(?:st|nd|rd|th)?(?:,)?[-.\s]+\d{2,4}|'
    r'\d{1,2}(?:st|nd|rd|th)?[-.\s]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:,)?[-.\s]+\d{2,4}|'
    # Text formats: January 1st, 2023; 1st of January, 2023
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{2,4}|'
    r'\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:,)?\s+\d{2,4}|'
    # Compact formats: YYYYMMDD, MMDDYYYY
    r'\b\d{8}\b|'
    # Age with time units that suggest DOB
    r'\b\d{1,3}\s*(?:year|yr|month|week|day|hour)s?\s*(?:of\s+age|old)\b'
    r')\b', re.IGNORECASE
)

# ID patterns
ID_PATTERN = re.compile(r'\b(?:\d[-]?){5,}\b|\b[A-Z]\d{5,}\b|\b\d{5,}[A-Z]?\b')
SSN_PATTERN = re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b')
PHONE_PATTERN = re.compile(
    r'(\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|'  # US/Canada: (123) 456-7890, 123-456-7890
    r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b)'  # International: +XX XXX XXXXXXX
)
ULTRASOUND_ID_PATTERN = re.compile(r'\b(?:\d{2}[-]){2}\d{2}[-]\d{6,}\b|\b\d{8}[_]\d{6,}\b')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
URL_PATTERN = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')

# Expanded non-sensitive technical terms
NON_SENSITIVE_TECHNICAL_TERMS = [
 
    # Imaging terminology
    'image', 'slice', 'frame', 'series', 'scan', 'sequence', 'protocol', 'cine', 'zoom',
    'thick', 'thickness', 'spacing', 'kernel', 'window', 'level', 'fov', 'field of view', 'matrix', 'recon',
    'reconstruction', 'table', 'position', 'rotation', 'tilt', 'pitch', 'detector', 'acquisition',
    
    # Technical parameters
    'kv', 'kvp', 'mas', 'ma', 'ctdi', 'dlp', 'contrast', 'delay', 'angle', 'fps', 'hz', 'fr',
    'db', 'mhz', 'thz', 'mm', 'cm', 'ml', 'cc', 'msec', 'sec', 'min', 'resolution',
    
    # Anatomical directions
    'left', 'right', 'anterior', 'posterior', 'superior', 'inferior', 'medial', 'lateral',
    'proximal', 'distal', 'caudal', 'cranial', 'sagittal', 'coronal', 'axial', 'oblique', 'transverse',
    'transaxial', 'longitudinal', 'obl', 'vol', 'avg', 'scout', 'topogram', 'localizer',
    
    # Report sections
    'archive', 'approved', 'verified', 'report', 'view', 'technique', 'finding', 'impression', 'history',
    'reason', 'procedure', 'clinical', 'indication', 'comparison', 'conclusion', 'recommendation',
    
    # Manufacturers
    'philips', 'ge', 'siemens', 'toshiba', 'canon', 'hologic', 'esaote', 'bk medical',
    'healthcare', 'medical systems', 'healthineers', 'fujifilm', 'hitachi', 'samsung', 'medtronic',
    'mindray', 'carestream', 'agfa', 'konica', 'minolta', 'picker', 'shimadzu',
    
    # System information
    'version', 'software', 'model', 'system', 'device', 'platform', 'workstation', 'pacs',
    'dicom', 'dcm', 'transfer', 'import', 'export', 'upload', 'download', 'processing',
    
    # Cardiac/ECG
    'curve', 'ecg', 'ekg', 'lead', 'bpm', 'hr', 'mi', 'tis', 'tic', 'qrs', 'st', 'pr',
    'systole', 'diastole', 'rhythm', 'wave', 'segment', 'interval',
    
    # Ultrasound
    'power', 'gain', 'depth', 'dynamic range', 'dr', 'map', 'filter', 'comp', 'probe',
    'transducer', 'mhz', 'harmonic', 'doppler', 'pulse', 'cw', 'pw', 'color', 'elastography',
    
    # MR sequences
    't1', 't2', 'pd', 'flair', 'dwi', 'adc', 'gd', 'gadolinium', 'fs', 'ir', 'gre', 'fse', 'se',
    'stir', 'mprage', 'spgr', 'fiesta', 'balance', 'ssfp', 'epi', 'swr', 'swan', 'mra', 'mrv',
    'dti', 'bold', 'perfusion', 'arterial spin labeling', 'asl', 'diffusion', 'spectroscopy',
    
    # CT
    'cta', 'ctv', 'noncontrast', 'non-contrast', 'ncct', 'helical', 'axial', 'spiral',
    'bone', 'soft tissue', 'lung', 'angio', 'arterial', 'venous', 'portal', 'delay',
    
    # Common modalities
    'xray', 'x-ray', 'radiograph', 'ct', 'mr', 'mri', 'us', 'ultrasound', 'sono',
    'pet', 'spect', 'fluoroscopy', 'fluoro', 'mammo', 'mammography', 'tomo', 'tomosynthesis',
    'dexa', 'dxa', 'bone density', 'nuclear', 'angiography', 'interventional',
    
    # Measurements and units - EXPANDED
    'measurement', 'dimension', 'distance', 'area', 'volume', 'diameter', 'radius',
    'length', 'width', 'height', 'depth', 'size', 'mean', 'average', 'std', 'standard deviation',
    'min', 'max', 'median', 'ratio', 'hounsfield', 'hu', 'suv', 'adc', 'apparent diffusion coefficient',
    'millimeter', 'millimeters', 'centimeter', 'centimeters', 'meter', 'meters',
    'milliliter', 'milliliters', 'cubic centimeter', 'cubic centimeters', 'cc', 'liter', 'liters',
    'gram', 'grams', 'kilogram', 'kilograms', 'milligram', 'milligrams',
    'second', 'seconds', 'millisecond', 'milliseconds', 'minute', 'minutes', 'hour', 'hours',
    'hertz', 'kilohertz', 'megahertz', 'gigahertz',
    'degree', 'degrees', 'radian', 'radians',
    'pixel', 'pixels', 'voxel', 'voxels',
    'beat', 'beats', 'bpm', 'mmhg', 'pascal', 'pascals', 'kpa',
    'density', 'pressure', 'force', 'velocity', 'acceleration', 'speed',
    'concentration', 'percentage', 'percent', 'ratio', 'fraction', 'cm', 'mm', 'lt','ml',
    
    # Time and positioning
    'temp', 'temperature', 'time', 'date', 'phase', 'pre', 'post', 'prone', 'supine',
    'decubitus', 'upright', 'erect', 'recumbent', 'duration', 'interval',
    
    # Research-specific terms
    'score', 'grade', 'stage', 'classification', 'category', 'type', 'group',
    'control', 'test', 'experiment', 'trial', 'study', 'cohort', 'sample',
    'p-value', 'p value', 'confidence interval', 'ci', 'odds ratio', 'or',
    'relative risk', 'rr', 'hazard ratio', 'hr', 'standard error', 'se',
    'standard deviation', 'sd', 'interquartile range', 'iqr',
    'mean', 'median', 'mode', 'average', 'variance', 'correlation',
    'sensitivity', 'specificity', 'precision', 'recall', 'accuracy',
    'positive predictive value', 'ppv', 'negative predictive value', 'npv',
    'area under curve', 'auc', 'receiver operating characteristic', 'roc',
    
    # Statistical terms
    'normal', 'gaussian', 'distribution', 'parametric', 'nonparametric',
    'anova', 't-test', 't test', 'wilcoxon', 'mann-whitney', 'mann whitney',
    'chi-square', 'chi square', 'fisher', 'regression', 'correlation',
    'covariate', 'variable', 'dependent', 'independent', 'predictor',
    'univariate', 'multivariate', 'logistic', 'linear', 'categorical',
    'continuous', 'discrete', 'nominal', 'ordinal', 'interval', 'ratio',
    
    # Disease/condition measures
    'grade', 'stage', 'score', 'scale', 'index', 'severity', 'classification',
    'tnm', 'metastasis', 'recurrence', 'remission', 'survival',
    'progression', 'progression-free', 'progression free',
    'disease-free', 'disease free', 'overall survival', 'os',
    
    # Lab values
    'hemoglobin', 'hgb', 'hematocrit', 'hct', 'platelet', 'plt',
    'white blood cell', 'wbc', 'red blood cell', 'rbc',
    'albumin', 'globulin', 'protein', 'glucose', 'hba1c',
    'creatinine', 'bun', 'gfr', 'egfr', 'alt', 'ast', 'alp',
    'ldl', 'hdl', 'triglyceride', 'cholesterol', 'sodium', 'potassium',
    'chloride', 'calcium', 'phosphate', 'magnesium',
    'inr', 'pt', 'ptt', 'troponin', 'ck', 'ck-mb', 'ck mb',
    'psa', 'cea', 'afp', 'ca-125', 'ca125', 'ca 125',
    
    # Common modalities
    'xray', 'x-ray', 'radiograph', 'ct', 'mr', 'mri', 'us', 'ultrasound', 'sono',
    'pet', 'spect', 'fluoroscopy', 'fluoro', 'mammo', 'mammography', 'tomo', 'tomosynthesis',
    'dexa', 'dxa', 'bone density', 'nuclear', 'angiography', 'interventional',
    
    # Measurements and units
    'measurement', 'dimension', 'distance', 'area', 'volume', 'diameter', 'radius',
    'length', 'width', 'height', 'depth', 'size', 'mean', 'average', 'std', 'standard deviation',
    'min', 'max', 'median', 'ratio', 'hounsfield', 'hu', 'suv', 'adc', 'apparent diffusion coefficient',
    
    # Time and positioning
    'temp', 'temperature', 'time', 'date', 'phase', 'pre', 'post', 'prone', 'supine',
    'decubitus', 'upright', 'erect', 'recumbent', 'duration', 'interval',
    
    # Common medical measurements
    'bp', 'blood pressure', 'heart rate', 'pulse', 'respiratory rate', 'temperature',
    'oxygen', 'saturation', 'spo2', 'height', 'weight', 'bmi'
]

# Names of common medical institutions (to avoid flagging these as person names)
COMMON_INSTITUTION_WORDS = [
    'hospital', 'medical center', 'clinic', 'healthcare', 'health', 'memorial', 
    'regional', 'university', 'community', 'general', 'center', 'institute',
    'foundation', 'associates', 'care', 'group', 'physicians', 'specialists',
    'department', 'division', 'school', 'college', 'laboratory', 'imaging',
    'radiology', 'medicine', 'surgery', 'oncology', 'cardiology', 'neurology'
]



class DicomAnonymizer:
    def __init__(self, ocr_languages=['en'], ocr_gpu=False, confidence_threshold=0.4):
        self.reader = easyocr.Reader(ocr_languages, gpu=ocr_gpu)
        self.confidence_threshold = confidence_threshold
        self.stats = {
            'processed_files': 0, 'skipped_files': 0, 'removed_tags': 0,
            'redacted_text_regions': 0, 'errors': 0
        }
        self.debug_ocr_prep_path = Path("debug_ocr_prep")
        if not self.debug_ocr_prep_path.exists():
             try: self.debug_ocr_prep_path.mkdir(parents=True, exist_ok=True)
             except Exception as e:
                 logger.warning(f"Could not create debug OCR prep directory {self.debug_ocr_prep_path}: {e}")
                 self.debug_ocr_prep_path = None

    def is_sensitive_text(self, text):
        if not text:
            return False
        
        text_stripped = text.strip()
        
        # Skip very short text unless it's a known sensitive abbreviation
        if len(text_stripped) < 3:
            if text_stripped.lower() not in ['id', 'mr', 'dr', 'ms', 'sx', 'ex', 'kv', 'ip', 'op']: 
                return False
        
        text_lower = text_stripped.lower()
        
        # DIRECTLY CHECK FOR MEASUREMENTS WITH COMMA OR PERIOD DECIMAL SEPARATORS
        # This will catch values like "1,06 cm" or "1.06 cm"
        measurement_pattern = re.compile(
            r'^(?:\d+(?:[.,]\d+)?|\d+)(?:\s*x\s*(?:\d+(?:[.,]\d+)?|\d+))*\s*(?:mm|cm|m|ml|cc|sec|min|msec|hz|kg|g|lb|°|deg|degree|bpm|mmHg)s?$', 
            re.IGNORECASE
        )
        if measurement_pattern.match(text_stripped):
            logger.debug(f"Measurement value detected (non-sensitive): '{text_stripped}'")
            return False
        
        # Check for non-sensitive technical terms - return False only if definitely non-sensitive
        for term in NON_SENSITIVE_TECHNICAL_TERMS:
            # Exact match
            if re.fullmatch(term, text_lower):
                return False
                
            # Term with numbers - NOW SUPPORTING BOTH COMMA AND PERIOD DECIMALS
            term_with_num_pattern = f"{term}\\s*\\d+(?:[.,]\\d+)*[a-z%]*"
            if re.fullmatch(term_with_num_pattern, text_lower, re.IGNORECASE):
                return False
                
            # Numbers with term - NOW SUPPORTING BOTH COMMA AND PERIOD DECIMALS
            num_with_term_pattern = f"\\d+(?:[.,]\\d+)*[a-z%]*\\s*{term}"
            if re.fullmatch(num_with_term_pattern, text_lower, re.IGNORECASE):
                return False
                
            # Single word exact match
            if len(text_stripped.split()) == 1 and term == text_lower:
                return False
        
        # Special handling for measurement values with European number format (comma as decimal separator)
        european_measurement_pattern = re.compile(
            r'^\d+,\d+\s*(?:mm|cm|m|ml|cc|sec|min|msec|hz|kg|g|lb|°|deg|degree|bpm|mmHg)s?$', 
            re.IGNORECASE
        )
        if european_measurement_pattern.match(text_stripped):
            logger.debug(f"European format measurement value detected (non-sensitive): '{text_stripped}'")
            return False
        
        # Short numbers are typically not sensitive
        if text_stripped.isdigit() and len(text_stripped) < 4:
            return False
            
        # Known technical identifiers (Version numbers, software versions, etc.)
        technical_id_pattern = re.compile(
            r'^v\d+(?:\.\d+)*$|'  # version numbers like v1.2.3
            r'^\d+(?:\.\d+)+$|'   # decimal numbers like 3.14.1
            r'^r\d+\.\d+$',       # release numbers like r2.0
            re.IGNORECASE
        )
        if technical_id_pattern.match(text_stripped):
            return False
        
        # Check for explicit sensitive patterns - if found, mark as sensitive
        if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in SENSITIVE_KEYWORDS_REFINED):
            logger.debug(f"Sensitive keyword found in: '{text_stripped}'")
            return True
            
        if DATE_PATTERN.search(text_stripped):
            logger.debug(f"Date pattern found in: '{text_stripped}'")
            return True
            
        if SSN_PATTERN.search(text_stripped):
            logger.debug(f"SSN pattern found in: '{text_stripped}'")
            return True
            
        if PHONE_PATTERN.search(text_stripped):
            logger.debug(f"Phone pattern found in: '{text_stripped}'")
            return True
            
        if ULTRASOUND_ID_PATTERN.search(text_stripped):
            logger.debug(f"Ultrasound ID pattern found in: '{text_stripped}'")
            return True
            
        if EMAIL_PATTERN.search(text_stripped):
            logger.debug(f"Email address found in: '{text_stripped}'")
            return True
        
        # For ID patterns: Use a less restrictive approach - if it looks like an ID, mark it sensitive
        id_match = ID_PATTERN.search(text_stripped)
        if id_match:
            # EXCEPTION: If it's a measurement with units, don't mark as sensitive
            if re.search(r'\d+(?:[.,]\d+)?\s*(?:mm|cm|m|ml|cc|sec|min|msec|hz|kg|g|lb)', text_stripped, re.IGNORECASE):
                return False
            logger.debug(f"Potential ID found in: '{text_stripped}'")
            return True
            
        # More measurement patterns to handle edge cases
        complex_measurement_pattern = re.compile(
            r'^(?:\d+(?:[.,]\d+)?|\d+)\s*(?:x|\*)\s*(?:\d+(?:[.,]\d+)?|\d+)\s*(?:mm|cm|m|ml|cc|sec|min|hz)s?$', 
            re.IGNORECASE
        )
        if complex_measurement_pattern.match(text_stripped):
            logger.debug(f"Complex measurement detected (non-sensitive): '{text_stripped}'")
            return False
        
        # DEFAULT CASE: If we got here, and the text contains letters, assume it might be sensitive
        if any(c.isalpha() for c in text_stripped):
            # Handle some known safe text patterns
            
            # Single lowercase word that's likely a common term
            if (len(text_stripped.split()) == 1 and 
                text_stripped.islower() and 
                len(text_stripped) <= 10 and 
                text_stripped.isalpha()):
                common_words = [
                    'left', 'right', 'top', 'bottom', 'front', 'back', 'upper', 'lower',
                    'inner', 'outer', 'anterior', 'posterior', 'medial', 'lateral',
                    'yes', 'no', 'true', 'false', 'none', 'normal', 'abnormal',
                    'view', 'scan', 'image', 'series', 'study'
                ]
                if text_lower in common_words:
                    return False
                    
            # By default, treat any text with letters as potentially sensitive
            logger.debug(f"Marking as sensitive by default (contains letters): '{text_stripped}'")
            return True
        
        # Numeric text not caught by earlier rules - could be sensitive
        if any(c.isdigit() for c in text_stripped):
            # Final check for measurements
            possible_measurement = re.match(r'^\d+(?:[.,]\d+)?$', text_stripped)
            if possible_measurement:
                logger.debug(f"Possible numeric value (allowing): '{text_stripped}'")
                return False
            logger.debug(f"Marking as sensitive by default (contains numbers): '{text_stripped}'")
            return True
            
        # If we somehow get here (text with no letters or numbers?), mark as non-sensitive
        return False
    def _get_preprocessed_versions(self, img_bgr_input, base_filename="unknown"):
        """
        Prepares image versions for OCR. Currently uses equalized grayscale.
        Can be extended to try multiple preprocessing strategies.
        """
        images_to_ocr = {}
        img_bgr_orig = img_bgr_input.copy()
        prep_name = "simple_equalized_hist_gray"
        try:
            gray_for_ocr = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2GRAY)
            gray_for_ocr = cv2.equalizeHist(gray_for_ocr) # Histogram equalization
            images_to_ocr[prep_name] = gray_for_ocr
            logger.info(f"Generated '{prep_name}' for OCR input.")
        except Exception as e:
            logger.error(f"Error in '{prep_name}' preprocessing for {base_filename}: {e}")
            logger.warning(f"Falling back to original BGR for OCR for {base_filename} due to preprocessing error.")
            images_to_ocr["original_bgr_fallback"] = img_bgr_orig.copy()

        if self.debug_ocr_prep_path:
            for name, img_data in images_to_ocr.items():
                try:
                    cv2.imwrite(str(self.debug_ocr_prep_path / f"{base_filename}_prep_FOR_OCR_{name}.png"), img_data)
                except Exception as e_save:
                    logger.error(f"Failed to save OCR input debug image {name} for {base_filename}: {e_save}")
            if images_to_ocr:
                 logger.info(f"Saved OCR input image(s) to {self.debug_ocr_prep_path} (base: {base_filename}).")
        return images_to_ocr

    def redact_burned_in_text(self, ds, base_filename_for_debug="unknown_dcm"):
        """
        This is the more ADVANCED redaction method that handles normalization,
        color preservation, and attempts to revert to original dtype.
        This version should address the quality and grayscale issues.
        """
        if not hasattr(ds, 'PixelData') or ds.PixelData is None or len(ds.PixelData) == 0:
            logger.warning(f"No pixel data in DICOM file {base_filename_for_debug}, skipping redaction."); return 0

        try:
            logger.info(f"Starting redaction for {base_filename_for_debug} (using ADVANCED pixel handling)...")
            # Keep a copy of the original pixel array and its properties
            original_pixel_array = ds.pixel_array.copy()
            original_dtype = original_pixel_array.dtype
            original_shape = original_pixel_array.shape
            original_samples_per_pixel = ds.get('SamplesPerPixel', 1)
            original_photometric_interpretation = ds.get('PhotometricInterpretation', '')

            pixel_min_orig = original_pixel_array.min()
            pixel_max_orig = original_pixel_array.max()
            logger.debug(f"Original pixel data: dtype={original_dtype}, shape={original_shape}, "
                         f"min={pixel_min_orig}, max={pixel_max_orig}, SPP={original_samples_per_pixel}, PI={original_photometric_interpretation}")


            # --- Step 1: Prepare initial BGR uint8 version from DICOM (base_bgr_image) ---
            # This base_bgr_image aims to represent the visual data for redaction.
            # It performs normalization if the original data is not uint8.
            img_uint8_initial = None
            if original_dtype != np.uint8:
                if pixel_max_orig > pixel_min_orig: # Avoid division by zero if range is 0
                    # Normalize to 0-255
                    img_norm = ((original_pixel_array - pixel_min_orig) / (pixel_max_orig - pixel_min_orig) * 255.0)
                    img_uint8_initial = img_norm.astype(np.uint8)
                    logger.debug(f"Normalized pixel data from {original_dtype} to uint8 (range {pixel_min_orig}-{pixel_max_orig} -> 0-255).")
                else: # Image is flat (all pixels same value)
                    # Convert the single value to an 8-bit equivalent
                    target_val_uint8 = int(pixel_min_orig) if np.issubdtype(original_dtype, np.integer) else 0
                    if pixel_max_orig != 0: # if not already 0
                        try:
                            target_val_uint8 = int((float(pixel_min_orig) / float(pixel_max_orig)) * 255.0) if pixel_max_orig != 0 else 0
                        except ZeroDivisionError: # Should be caught by pixel_max_orig > pixel_min_orig but as safety
                            target_val_uint8 = 0

                    img_uint8_initial = np.full(original_shape, np.clip(target_val_uint8,0,255), dtype=np.uint8)
                    logger.debug(f"Image is flat. Converted to uint8 with value {target_val_uint8}.")
            else: # Already uint8
                img_uint8_initial = original_pixel_array.copy()
                logger.debug("Original pixel data is already uint8.")

            base_bgr_image = None
            if len(img_uint8_initial.shape) == 2: # Grayscale
                base_bgr_image = cv2.cvtColor(img_uint8_initial, cv2.COLOR_GRAY2BGR)
            elif len(img_uint8_initial.shape) == 3 and img_uint8_initial.shape[2] == 1: # Grayscale with channel dim
                base_bgr_image = cv2.cvtColor(img_uint8_initial, cv2.COLOR_GRAY2BGR)
            elif len(img_uint8_initial.shape) == 3 and img_uint8_initial.shape[2] == 3: # RGB/BGR 3-channel
                base_bgr_image = img_uint8_initial.copy()
                # If pydicom provides RGB and original PI was RGB, convert to BGR for OpenCV processing
                if original_photometric_interpretation == "RGB":
                    logger.debug("Original PI is RGB, converting pixel data from RGB to BGR for OpenCV.")
                    base_bgr_image = cv2.cvtColor(base_bgr_image, cv2.COLOR_RGB2BGR)
            elif len(img_uint8_initial.shape) == 3 and img_uint8_initial.shape[2] == 4: # RGBA/BGRA
                logger.debug("Input has 4 channels, attempting to convert to BGR.")
                if original_photometric_interpretation.startswith("RGBA"): # e.g. RGBA8888
                    base_bgr_image = cv2.cvtColor(img_uint8_initial, cv2.COLOR_RGBA2BGR)
                else: # Assume BGRA or other 4-channel, try BGRA2BGR
                    base_bgr_image = cv2.cvtColor(img_uint8_initial, cv2.COLOR_BGRA2BGR)
            else:
                logger.error(f"❌ Unhandled image shape/channels for OCR: {img_uint8_initial.shape} for {base_filename_for_debug}"); return 0

            if self.debug_ocr_prep_path: # Save the BGR image that redactions will be drawn on
                try: cv2.imwrite(str(self.debug_ocr_prep_path / f"{base_filename_for_debug}_base_bgr_for_redaction_ADVANCED.png"), base_bgr_image)
                except Exception as e_save: logger.error(f"Error saving base_bgr_for_redaction_ADVANCED.png: {e_save}")

            # --- Step 2: Get preprocessed version(s) specifically FOR OCR ---
            images_for_ocr_dict = self._get_preprocessed_versions(base_bgr_image, base_filename=base_filename_for_debug)

            # --- Step 3: Run OCR on the preprocessed image(s) ---
            all_ocr_results = []
            for ocr_input_name, image_sent_to_ocr in images_for_ocr_dict.items():
                if image_sent_to_ocr is None or image_sent_to_ocr.size == 0: continue
                logger.info(f"Running OCR on: {ocr_input_name} for {base_filename_for_debug}")
                try:
                    ocr_ready_img = np.ascontiguousarray(image_sent_to_ocr)
                    current_raw_results = self.reader.readtext(ocr_ready_img, detail=1, paragraph=False)
                    logger.info(f"Found {len(current_raw_results)} texts in '{ocr_input_name}'.")
                    for res_item in current_raw_results:
                        all_ocr_results.append((res_item[0], res_item[1], res_item[2], ocr_input_name)) # bbox, text, conf, source
                except Exception as ocr_exc: logger.error(f"OCR error for {ocr_input_name} on {base_filename_for_debug}: {ocr_exc}", exc_info=True)

            # Simple de-duplication if multiple OCR preps were used (not active with current _get_preprocessed_versions)
            final_detections_to_process = all_ocr_results
            if len(images_for_ocr_dict) > 1 and all_ocr_results:
                temp_map = {} # Using a simple dict to overwrite with higher confidence for similar text/location
                for bbox_pts, text, conf, src in all_ocr_results:
                    # Key by text and approximate top-left corner (divided by 10 for tolerance)
                    key = (text, int(bbox_pts[0][0]/10), int(bbox_pts[0][1]/10))
                    if key not in temp_map or conf > temp_map[key][2]: # if new or higher confidence
                        temp_map[key] = (bbox_pts, text, conf, src)
                final_detections_to_process = list(temp_map.values())
                logger.info(f"Deduplicated OCR results from {len(all_ocr_results)} to {len(final_detections_to_process)}")

            logger.info(f"Processing {len(final_detections_to_process)} OCR detections for {base_filename_for_debug}")

            # --- Step 4: Draw redactions on a copy of the base_bgr_image ---
            redacted_count = 0
            bgr_image_with_redactions = base_bgr_image.copy() # Redactions are drawn on this BGR uint8 image

            for i, detection_item in enumerate(final_detections_to_process):
                bbox_points, text, conf, source_name = detection_item # Unpack
                if not (isinstance(bbox_points, list) and len(bbox_points) == 4 and all(isinstance(p, list) and len(p) == 2 for p in bbox_points)):
                    logger.warning(f"Skipping invalid bbox_points for text '{text}' from {source_name}"); continue

                logger.info(f"OCR Detection #{i+1}: '{text}', Confidence={conf:.3f} (from {source_name})")
                if conf < self.confidence_threshold:
                    logger.info(f"  ↳ SKIPPED: Low conf {conf:.3f} < threshold {self.confidence_threshold:.3f}"); continue

                is_sens = self.is_sensitive_text(text)
                logger.info(f"  ↳ Sensitivity Check for '{text}': {is_sens}")

                if is_sens:
                    all_x = [p[0] for p in bbox_points]; all_y = [p[1] for p in bbox_points]
                    top_left_rect = (min(all_x), min(all_y)); bottom_right_rect = (max(all_x), max(all_y))
                    top_left_int = tuple(map(int, top_left_rect)); bottom_right_int = tuple(map(int, bottom_right_rect))

                    cv2.rectangle(bgr_image_with_redactions, top_left_int, bottom_right_int, (0,0,0), -1) # Black
                    redacted_count += 1
                    logger.info(f"  ↳ ✅ REDACTED '{text}' with BLACK rectangle.")

            # --- Step 5: Prepare final pixel data for DICOM, preserving original color format and attempting to preserve dtype/range ---
            if redacted_count > 0:
                logger.info(f"Total regions redacted: {redacted_count} for {base_filename_for_debug}")
                if self.debug_ocr_prep_path:
                     try: cv2.imwrite(str(self.debug_ocr_prep_path / f"{base_filename_for_debug}_redacted_bgr_PRE_FINAL_ADVANCED.png"), bgr_image_with_redactions)
                     except Exception as e_save: logger.error(f"Error saving redacted_bgr_PRE_FINAL_ADVANCED.png: {e_save}")

                # This is the BGR uint8 image with redactions
                processed_uint8_for_conversion = bgr_image_with_redactions

                # Convert back to original color format (e.g., grayscale if original was grayscale)
                # And then scale back to original dtype and range if original was not uint8
                img_final_for_dicom = None

                if original_samples_per_pixel == 3: # Original was color
                    # `processed_uint8_for_conversion` is already BGR uint8.
                    # If original PI was RGB, convert BGR to RGB.
                    if original_photometric_interpretation == "RGB":
                        final_color_uint8 = cv2.cvtColor(processed_uint8_for_conversion, cv2.COLOR_BGR2RGB)
                        logger.info("Converted redacted BGR to RGB for DICOM storage (original PI was RGB).")
                    else: # Assume original was BGR or some other 3-channel format that pydicom gave as BGR-like
                        final_color_uint8 = processed_uint8_for_conversion
                        logger.info("Kept redacted image as BGR-like for color DICOM storage.")

                    if original_dtype != np.uint8: # Need to scale back
                        img_float = final_color_uint8.astype(np.float32) / 255.0
                        if pixel_max_orig > pixel_min_orig:
                            img_scaled = img_float * (pixel_max_orig - pixel_min_orig) + pixel_min_orig
                        else: # Flat image
                            img_scaled = np.full(final_color_uint8.shape, pixel_min_orig, dtype=np.float32)
                        img_final_for_dicom = img_scaled.astype(original_dtype)
                        logger.info(f"Scaled redacted color uint8 back to original dtype {original_dtype} and range.")
                    else: # Original was uint8 color
                        img_final_for_dicom = final_color_uint8.astype(np.uint8)
                else: # Original was grayscale (SamplesPerPixel=1 or not specified)
                    final_gray_uint8 = cv2.cvtColor(processed_uint8_for_conversion, cv2.COLOR_BGR2GRAY)
                    logger.info("Converting redacted BGR to Grayscale for final DICOM output.")

                    if original_dtype != np.uint8: # Need to scale back
                        img_float = final_gray_uint8.astype(np.float32) / 255.0
                        if pixel_max_orig > pixel_min_orig:
                            img_scaled = img_float * (pixel_max_orig - pixel_min_orig) + pixel_min_orig
                        else: # Flat image
                            img_scaled = np.full(final_gray_uint8.shape, pixel_min_orig, dtype=np.float32)
                        img_final_for_dicom = img_scaled.astype(original_dtype)
                        logger.info(f"Scaled redacted grayscale uint8 back to original dtype {original_dtype} and range.")
                    else: # Original was uint8 grayscale
                        img_final_for_dicom = final_gray_uint8.astype(np.uint8)

                # Update DICOM dataset with the new pixel data and related tags
                ds.PixelData = img_final_for_dicom.tobytes()
                ds.Rows, ds.Columns = img_final_for_dicom.shape[:2] # Works for grayscale (H,W) and color (H,W,C)

                # Update pixel data related tags to reflect the (potentially modified) final image
                ds.SamplesPerPixel = original_samples_per_pixel # Should match original
                ds.PhotometricInterpretation = original_photometric_interpretation # Should match original

                ds.BitsAllocated = img_final_for_dicom.dtype.itemsize * 8
                ds.BitsStored = ds.BitsAllocated # Common practice, could be ds.HighBit + 1 if specified
                ds.HighBit = ds.BitsStored - 1
                ds.PixelRepresentation = 1 if np.issubdtype(img_final_for_dicom.dtype, np.signedinteger) else 0

                if ds.SamplesPerPixel == 3:
                    ds.PlanarConfiguration = 0 # Common for interleaved color
                elif hasattr(ds, 'PlanarConfiguration'): # Grayscale does not have PlanarConfiguration
                    del ds.PlanarConfiguration

                if hasattr(ds, 'BurnedInAnnotation'): ds.BurnedInAnnotation = 'NO'
                self.stats['redacted_text_regions'] += redacted_count
                logger.info("Pixel data updated with redactions, attempting to preserve original characteristics.")
            else:
                logger.info(f"No sensitive text redacted for {base_filename_for_debug} using advanced OCR pixel handling.")

            return redacted_count
        except Exception as e:
            logger.error(f"❌ Error in redact_burned_in_text (ADVANCED) for {base_filename_for_debug}: {e}", exc_info=True)
            self.stats['errors'] += 1
            return 0


    def anonymize_dataset(self, ds, keep_uids=False):
        ds_anon = ds.copy() # Work on a copy
        tags_removed = 0
        for tag_group, tag_element in SENSITIVE_TAGS:
            tag = (tag_group, tag_element)
            if tag in ds_anon:
                try:
                    tag_name = ds_anon[tag].name
                    tag_value_str = str(ds_anon[tag].value)
                    if "Patient" in tag_name and "Date" not in tag_name and len(tag_value_str) > 1 and tag_value_str.strip():
                        masked_value = tag_value_str[0] + "*" * (len(tag_value_str) - 2) + tag_value_str[-1] if len(tag_value_str) > 2 else "*"
                    else:
                        masked_value = tag_value_str # Or just "REDACTED"
                    logger.info(f"❌ Removing tag {tag_name} ({hex(tag_group)},{hex(tag_element)}): '{masked_value}'")
                    del ds_anon[tag]
                    tags_removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove/log tag ({hex(tag_group)},{hex(tag_element)}): {e}")

        ds_anon.remove_private_tags()
        logger.info("Removed private tags.")

        if not keep_uids:
            logger.info("Generating new UIDs...")
            uid_tags_to_regenerate = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'FrameOfReferenceUID']
            # Some UIDs are conditional (e.g., RelatedFrameOfReferenceUID if FrameOfReferenceUID is changed)
            # For simplicity, only common ones are handled here.
            for uid_tag_name in uid_tags_to_regenerate:
                if uid_tag_name in ds_anon:
                    original_uid = ds_anon.data_element(uid_tag_name).value
                    new_uid = generate_uid()
                    setattr(ds_anon, uid_tag_name, new_uid)
                    logger.debug(f"  Replaced {uid_tag_name}: {original_uid} -> {new_uid}")

            # Update MediaStorageSOPInstanceUID if SOPInstanceUID was changed
            if hasattr(ds_anon, 'file_meta') and 'SOPInstanceUID' in ds_anon and \
               'MediaStorageSOPInstanceUID' in ds_anon.file_meta:
                if ds_anon.file_meta.MediaStorageSOPInstanceUID != ds_anon.SOPInstanceUID:
                    logger.debug(f"  Updating MediaStorageSOPInstanceUID to match new SOPInstanceUID: {ds_anon.SOPInstanceUID}")
                    ds_anon.file_meta.MediaStorageSOPInstanceUID = ds_anon.SOPInstanceUID
        else:
            logger.info("Keeping original UIDs.")

        self.stats['removed_tags'] += tags_removed
        return ds_anon

    def anonymize_file(self, input_path_str, output_path_str, redact_overlays=True, keep_uids=False):
        input_path = Path(input_path_str)
        output_path = Path(output_path_str)
        base_filename = input_path.stem
        try:
            logger.info(f"📄 Processing: {input_path}")
            logger.info(f"📤 Output to: {output_path}")

            try:
                # Read the DICOM file. force=True can help with some problematic files.
                ds = pydicom.dcmread(str(input_path), force=True)
            except Exception as e:
                logger.error(f"❌ Error reading {input_path}: {e}")
                self.stats['errors'] += 1
                self.stats['skipped_files'] += 1
                return False

            # Create a deep copy for anonymization to leave original ds untouched if needed elsewhere
            # Though anonymize_dataset already makes a copy, this is safer if ds is used before that.
            ds_to_anonymize = copy.deepcopy(ds)

            # Step 1: Anonymize metadata
            ds_anon = self.anonymize_dataset(ds_to_anonymize, keep_uids)

            # Step 2: Redact burned-in text from pixel data if requested and possible
            has_px_data = hasattr(ds_anon, 'PixelData') and ds_anon.PixelData is not None and len(ds_anon.PixelData) > 0
            if redact_overlays and has_px_data:
                logger.info(f"Proceeding with pixel data redaction for {base_filename}.")
                self.redact_burned_in_text(ds_anon, base_filename_for_debug=base_filename)
                # Pixel data modification often means data is uncompressed.
                # Set TransferSyntaxUID to ExplicitVRLittleEndian.
                if hasattr(ds_anon, 'file_meta'):
                    ds_anon.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                else:
                    # This case should ideally not happen if reading a valid DICOM file
                    logger.warning(f"File {base_filename} lacks file_meta. Creating a new one.")
                    ds_anon.file_meta = pydicom.Dataset()
                    ds_anon.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            elif not has_px_data:
                logger.warning(f"⚠️ No pixel data in {input_path} or pixel data is empty - OCR related redaction skipped.")
            elif not redact_overlays:
                logger.info(f"Skipping pixel data redaction as per request for {base_filename}.")


            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Update content date/time to reflect anonymization time
            now = datetime.now()
            ds_anon.ContentDate = now.strftime('%Y%m%d')
            ds_anon.ContentTime = now.strftime('%H%M%S.%f')[:14] # Max 14 chars for TM VR

            # Ensure essential file_meta attributes are present and consistent
            if not hasattr(ds_anon, 'file_meta'): # Should have been created if missing and redaction happened
                ds_anon.file_meta = pydicom.Dataset()

            # SOP Class UID - should exist in a valid DICOM
            if 'SOPClassUID' in ds_anon:
                 ds_anon.file_meta.MediaStorageSOPClassUID = ds_anon.SOPClassUID
            elif 'MediaStorageSOPClassUID' not in ds_anon.file_meta : # if SOPClassUID also missing
                 logger.warning(f"SOPClassUID missing for {base_filename}, using default Secondary Capture. This may not be appropriate.")
                 ds_anon.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
                 if 'SOPClassUID' not in ds_anon: ds_anon.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage # Also set in dataset

            # SOP Instance UID - ensure it matches if file_meta exists
            if 'SOPInstanceUID' in ds_anon:
                ds_anon.file_meta.MediaStorageSOPInstanceUID = ds_anon.SOPInstanceUID
            elif 'MediaStorageSOPInstanceUID' not in ds_anon.file_meta: # if SOPInstanceUID also missing
                new_sop_instance_uid = generate_uid() # Should have been generated if keep_uids=False
                ds_anon.file_meta.MediaStorageSOPInstanceUID = new_sop_instance_uid
                if 'SOPInstanceUID' not in ds_anon: ds_anon.SOPInstanceUID = new_sop_instance_uid

            if 'TransferSyntaxUID' not in ds_anon.file_meta:
                # If redaction happened, it would have been set. If not, preserve original or default.
                original_ts = ds.file_meta.TransferSyntaxUID if hasattr(ds, 'file_meta') and 'TransferSyntaxUID' in ds.file_meta else ExplicitVRLittleEndian
                ds_anon.file_meta.TransferSyntaxUID = original_ts
                if original_ts == ExplicitVRLittleEndian:
                    logger.debug(f"Using/Setting TransferSyntaxUID to ExplicitVRLittleEndian for {base_filename}")
                else:
                    logger.debug(f"Preserving original TransferSyntaxUID {original_ts} for {base_filename}")


            if 'ImplementationClassUID' not in ds_anon.file_meta:
                ds_anon.file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
            if 'ImplementationVersionName' not in ds_anon.file_meta:
                ds_anon.file_meta.ImplementationVersionName = f"PYDICOM_ANON_V2 {pydicom.__version__}" # Minor version bump in name

            try:
                # Ensure is_little_endian and is_implicit_VR match the TransferSyntaxUID
                # This is critical if TransferSyntaxUID was changed (e.g., to ExplicitVRLittleEndian after redaction)
                current_ts_uid = pydicom.uid.UID(ds_anon.file_meta.TransferSyntaxUID)
                ds_anon.is_little_endian = current_ts_uid.is_little_endian
                ds_anon.is_implicit_VR = current_ts_uid.is_implicit_VR

                # write_like_original=False is generally safer after significant modifications like pixel data changes.
                ds_anon.save_as(str(output_path), write_like_original=False)
                logger.info(f"✅ Saved: {output_path}")
                self.stats['processed_files'] += 1
                return True
            except Exception as e_save:
                logger.error(f"❌ Error saving {output_path} (write_like_original=False): {e_save}")
                logger.warning("⚠️ Fallback: Attempting save with dcmwrite and forced ExplicitVRLittleEndian.")
                try:
                    # Force ExplicitVRLittleEndian as a fallback save strategy
                    ds_anon.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    ds_anon.is_little_endian = True
                    ds_anon.is_implicit_VR = False
                    pydicom.dcmwrite(str(output_path), ds_anon, write_like_original=False)
                    logger.info(f"✅ Saved with dcmwrite (forced ExplicitVRLittleEndian): {output_path}")
                    self.stats['processed_files'] += 1
                    return True
                except Exception as e2_save:
                    logger.error(f"❌ Final save error for {output_path}: {e2_save}", exc_info=True)
                    self.stats['errors'] += 1
                    self.stats['skipped_files'] += 1
                    return False

        except Exception as e_main:
            logger.error(f"❌ Unhandled error processing {input_path}: {e_main}", exc_info=True)
            self.stats['errors'] += 1
            self.stats['skipped_files'] += 1
            return False

    def anonymize_directory(self, input_dir_str, output_dir_str, recursive=True, file_pattern='*.dcm',
                           redact_overlays=True, keep_uids=False):
        self.stats = {k: 0 for k in self.stats} # Reset stats for directory run
        input_dir = Path(input_dir_str)
        output_dir = Path(output_dir_str)

        if not input_dir.is_dir():
            logger.error(f"Input directory does not exist or is not a directory: {input_dir}")
            return self.stats

        output_dir.mkdir(parents=True, exist_ok=True)

        files_to_process = []
        if recursive:
            files_to_process = list(input_dir.rglob(file_pattern))
        else:
            files_to_process = list(input_dir.glob(file_pattern))

        logger.info(f"Found {len(files_to_process)} files in {input_dir} (pattern: '{file_pattern}', recursive: {recursive})")

        for file_path in filter(Path.is_file, files_to_process): # Ensure it's a file
            relative_path = file_path.relative_to(input_dir)
            output_file_path = output_dir / relative_path
            output_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output subdirectories exist
            self.anonymize_file(str(file_path), str(output_file_path), redact_overlays, keep_uids)

        logger.info("📊 Directory anonymization complete. Stats:")
        for k,v in self.stats.items():
            logger.info(f"  {k.replace('_',' ').capitalize()}: {v}")
        return self.stats


# Main execution block
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='DICOM Anonymizer with Advanced Pixel Handling')
    parser.add_argument('--input', required=True, help='Input DICOM file or directory')
    parser.add_argument('--output', required=True, help='Output DICOM file or directory')
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--pattern', default='*.dcm', help='File pattern (e.g., "*.dcm", "*.dicom")')
    parser.add_argument('--keep-uids', action='store_true', help='Keep original UIDs (Study, Series, SOP Instance)')
    parser.add_argument('--no-redact', action='store_true', help='Skip burned-in text redaction entirely')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'], help='Logging level')
    parser.add_argument('--confidence', type=float, default=0.4, help='OCR confidence threshold for redaction (0.0-1.0)')
    parser.add_argument('--ocr-gpu', action='store_true', help='Use GPU for EasyOCR if available.')


    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    anonymizer = DicomAnonymizer(ocr_gpu=args.ocr_gpu, confidence_threshold=args.confidence)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        exit(1)

    if input_path.is_dir():
        anonymizer.anonymize_directory(str(input_path), str(output_path),
                                       recursive=args.recursive,
                                       file_pattern=args.pattern,
                                       redact_overlays=not args.no_redact,
                                       keep_uids=args.keep_uids)
    elif input_path.is_file():
        if output_path.is_dir() or \
           (not output_path.suffix and not output_path.exists()) or \
           (output_path.exists() and output_path.is_dir()): # More robust check if output_path is a directory
            output_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            output_file_path = output_path / input_path.name
        else: # Assume output_path is a full file path
            output_file_path = output_path
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

        anonymizer.anonymize_file(str(input_path), str(output_file_path),
                                  redact_overlays=not args.no_redact,
                                  keep_uids=args.keep_uids)
    else:
        logger.error(f"Input path is not a valid file or directory: {input_path}")
        exit(1)

    logger.info("Anonymization process finished.")