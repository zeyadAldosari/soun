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
import json
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

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

class DicomAnonymizer:
    def __init__(self, ocr_languages=['en'], ocr_gpu=False, confidence_threshold=0.4, llm_model="gpt-4o-mini"):
        self.reader = easyocr.Reader(ocr_languages, gpu=ocr_gpu)
        self.confidence_threshold = confidence_threshold
        self.llm_model = llm_model
        self.stats = {
            'processed_files': 0, 'skipped_files': 0, 'removed_tags': 0,
            'redacted_text_regions': 0, 'errors': 0, 'llm_calls': 0
        }
        self.debug_ocr_prep_path = Path("debug_ocr_prep")
        if not self.debug_ocr_prep_path.exists():
             try: self.debug_ocr_prep_path.mkdir(parents=True, exist_ok=True)
             except Exception as e:
                 logger.warning(f"Could not create debug OCR prep directory {self.debug_ocr_prep_path}: {e}")
                 self.debug_ocr_prep_path = None

    def classify_texts_with_llm(self, texts):
        """
        Use OpenAI's LLM to classify texts as sensitive or not
        
        Args:
            texts: List of text strings detected by OCR
        
        Returns:
            Dictionary mapping each text to a boolean (True if sensitive, False if not)
        """
        if not texts:
            return {}
        
        # Remove duplicate texts to minimize API usage
        unique_texts = list(set(texts))
        
        # Prepare the prompt
        prompt = """
        You are an expert in medical data privacy and HIPAA compliance with an EXTREMELY CONSERVATIVE approach to data protection. Below is a list of text elements detected in a medical image. For each text, determine if it contains sensitive or personally identifiable information that should be redacted.
        
        YOUR PRIMARY DIRECTIVE IS TO ERR ON THE SIDE OF CAUTION. When in doubt, ALWAYS classify text as sensitive.
        
        Sensitive information includes (this list is NOT exhaustive):
        - ANY names (patients, doctors, nurses, technicians, etc.)
        - ALL identifiers (patient IDs, MRNs, chart numbers, etc.)
        - ANY dates (birth dates, exam dates, appointment dates, etc.)
        - ALL numeric sequences that could be identifiers
        - ANY contact information (phone, email, address, etc.)
        - Institution names, hospital names, clinic names
        - Location information of any kind (city, state, zip, etc.)
        - Device serial numbers, unique equipment identifiers
        - Any text that includes a person's age or birth year
        - Study or exam identifiers, accession numbers
        - Insurance information, billing codes if patient-specific
        - ANY alphanumeric code that isn't clearly a technical parameter
        - ANY text containing both letters and numbers (could be an ID)
        - Room numbers, bed numbers, department identifiers
        - ANY text that could potentially identify a specific individual, facility, or encounter
        
        Non-sensitive information includes ONLY:
        - Pure anatomical terms (heart, lung, brain, etc.)
        - Standard measurement values with units (e.g., "10.5 cm", "150 mmHg")
        - Technical imaging parameters (e.g., "120kV", "150mA")
        - Generic directional indicators (LEFT, RIGHT, ANTERIOR, etc.)
        - Common medical terminology without specifics
        
        IMPORTANT: If you have ANY doubt about whether text is sensitive, CLASSIFY IT AS SENSITIVE.
        
        For each text element, respond with either "sensitive" or "not_sensitive". Format your response as a JSON object where the keys are the exact text strings and the values are boolean (true for sensitive, false for not sensitive).
        
        Text elements to classify:
        """
        
        # Add each text item
        for text in unique_texts:
            prompt += f"\n- \"{text}\""
        
        prompt += "\n\nRespond with a JSON object only."
        
        try:
            # Call the OpenAI API
            logger.info(f"Sending {len(unique_texts)} text elements to OpenAI for classification")
            self.stats['llm_calls'] += 1
            
            response = openai.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a medical privacy expert that only responds with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Log the results
            sensitive_count = sum(1 for value in result.values() if value is True)
            logger.info(f"LLM classified {sensitive_count} out of {len(result)} text elements as sensitive")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # In case of error, classify all texts as sensitive to be safe
            return {text: True for text in unique_texts}

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
        Advanced redaction method that uses an LLM to determine which text is sensitive.
        """
        if not hasattr(ds, 'PixelData') or ds.PixelData is None or len(ds.PixelData) == 0:
            logger.warning(f"No pixel data in DICOM file {base_filename_for_debug}, skipping redaction."); return 0

        try:
            logger.info(f"Starting redaction for {base_filename_for_debug} (using LLM-based pixel handling)...")
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
                try: cv2.imwrite(str(self.debug_ocr_prep_path / f"{base_filename_for_debug}_base_bgr_for_redaction_LLM.png"), base_bgr_image)
                except Exception as e_save: logger.error(f"Error saving base_bgr_for_redaction_LLM.png: {e_save}")

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

            # Simple de-duplication if multiple OCR preps were used
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

            # --- Step 4: Filter OCR results by confidence threshold ---
            filtered_detections = []
            all_texts = []
            bbox_mapping = {}  # Map text to bbox and confidence
            
            for i, detection_item in enumerate(final_detections_to_process):
                bbox_points, text, conf, source_name = detection_item
                logger.info(f"OCR Detection #{i+1}: '{text}', Confidence={conf:.3f} (from {source_name})")
                
                if conf >= self.confidence_threshold:
                    filtered_detections.append(detection_item)
                    all_texts.append(text)
                    bbox_mapping[text] = (bbox_points, conf)
                else:
                    logger.info(f"  ↳ SKIPPED: Low conf {conf:.3f} < threshold {self.confidence_threshold:.3f}")
            
            # --- Step 5: Use LLM to classify all texts at once ---
            if not all_texts:
                logger.info(f"No texts with sufficient confidence found for {base_filename_for_debug}")
                return 0
                
            sensitivity_map = self.classify_texts_with_llm(all_texts)
            
            # --- Step 6: Apply redactions based on LLM classification ---
            redacted_count = 0
            bgr_image_with_redactions = base_bgr_image.copy()
            
            for text, is_sensitive in sensitivity_map.items():
                if text in bbox_mapping:
                    bbox_points, conf = bbox_mapping[text]
                    logger.info(f"LLM classification for '{text}': {'SENSITIVE' if is_sensitive else 'NOT SENSITIVE'}")
                    
                    if is_sensitive:
                        all_x = [p[0] for p in bbox_points]
                        all_y = [p[1] for p in bbox_points]
                        top_left_rect = (min(all_x), min(all_y))
                        bottom_right_rect = (max(all_x), max(all_y))
                        top_left_int = tuple(map(int, top_left_rect))
                        bottom_right_int = tuple(map(int, bottom_right_rect))
                        
                        cv2.rectangle(bgr_image_with_redactions, top_left_int, bottom_right_int, (0,0,0), -1)  # Black
                        redacted_count += 1
                        logger.info(f"  ↳ ✅ REDACTED '{text}' with BLACK rectangle.")

            # --- Step 7: Prepare final pixel data for DICOM, preserving original color format and dtype ---
            if redacted_count > 0:
                logger.info(f"Total regions redacted: {redacted_count} for {base_filename_for_debug}")
                if self.debug_ocr_prep_path:
                     try: cv2.imwrite(str(self.debug_ocr_prep_path / f"{base_filename_for_debug}_redacted_bgr_PRE_FINAL_LLM.png"), bgr_image_with_redactions)
                     except Exception as e_save: logger.error(f"Error saving redacted_bgr_PRE_FINAL_LLM.png: {e_save}")

                # This is the BGR uint8 image with redactions
                processed_uint8_for_conversion = bgr_image_with_redactions

                # Convert back to original color format and dtype
                img_final_for_dicom = None

                if original_samples_per_pixel == 3:  # Original was color
                    # If original PI was RGB, convert BGR to RGB
                    if original_photometric_interpretation == "RGB":
                        final_color_uint8 = cv2.cvtColor(processed_uint8_for_conversion, cv2.COLOR_BGR2RGB)
                        logger.info("Converted redacted BGR to RGB for DICOM storage (original PI was RGB).")
                    else:  # Assume original was BGR or similar
                        final_color_uint8 = processed_uint8_for_conversion
                        logger.info("Kept redacted image as BGR-like for color DICOM storage.")

                    if original_dtype != np.uint8:  # Need to scale back
                        img_float = final_color_uint8.astype(np.float32) / 255.0
                        if pixel_max_orig > pixel_min_orig:
                            img_scaled = img_float * (pixel_max_orig - pixel_min_orig) + pixel_min_orig
                        else:  # Flat image
                            img_scaled = np.full(final_color_uint8.shape, pixel_min_orig, dtype=np.float32)
                        img_final_for_dicom = img_scaled.astype(original_dtype)
                        logger.info(f"Scaled redacted color uint8 back to original dtype {original_dtype} and range.")
                    else:  # Original was uint8 color
                        img_final_for_dicom = final_color_uint8.astype(np.uint8)
                else:  # Original was grayscale (SamplesPerPixel=1 or not specified)
                    final_gray_uint8 = cv2.cvtColor(processed_uint8_for_conversion, cv2.COLOR_BGR2GRAY)
                    logger.info("Converting redacted BGR to Grayscale for final DICOM output.")

                    if original_dtype != np.uint8:  # Need to scale back
                        img_float = final_gray_uint8.astype(np.float32) / 255.0
                        if pixel_max_orig > pixel_min_orig:
                            img_scaled = img_float * (pixel_max_orig - pixel_min_orig) + pixel_min_orig
                        else:  # Flat image
                            img_scaled = np.full(final_gray_uint8.shape, pixel_min_orig, dtype=np.float32)
                        img_final_for_dicom = img_scaled.astype(original_dtype)
                        logger.info(f"Scaled redacted grayscale uint8 back to original dtype {original_dtype} and range.")
                    else:  # Original was uint8 grayscale
                        img_final_for_dicom = final_gray_uint8.astype(np.uint8)

                # Update DICOM dataset with the new pixel data and related tags
                ds.PixelData = img_final_for_dicom.tobytes()
                ds.Rows, ds.Columns = img_final_for_dicom.shape[:2]  # Works for grayscale (H,W) and color (H,W,C)

                # Update pixel data related tags to reflect the (potentially modified) final image
                ds.SamplesPerPixel = original_samples_per_pixel  # Should match original
                ds.PhotometricInterpretation = original_photometric_interpretation  # Should match original

                ds.BitsAllocated = img_final_for_dicom.dtype.itemsize * 8
                ds.BitsStored = ds.BitsAllocated  # Common practice, could be ds.HighBit + 1 if specified
                ds.HighBit = ds.BitsStored - 1
                ds.PixelRepresentation = 1 if np.issubdtype(img_final_for_dicom.dtype, np.signedinteger) else 0

                if ds.SamplesPerPixel == 3:
                    ds.PlanarConfiguration = 0  # Common for interleaved color
                elif hasattr(ds, 'PlanarConfiguration'):  # Grayscale does not have PlanarConfiguration
                    del ds.PlanarConfiguration

                if hasattr(ds, 'BurnedInAnnotation'): ds.BurnedInAnnotation = 'NO'
                self.stats['redacted_text_regions'] += redacted_count
                logger.info("Pixel data updated with redactions, attempting to preserve original characteristics.")
            else:
                logger.info(f"No sensitive text redacted for {base_filename_for_debug} using LLM-based OCR pixel handling.")

            return redacted_count
        except Exception as e:
            logger.error(f"❌ Error in redact_burned_in_text (LLM) for {base_filename_for_debug}: {e}", exc_info=True)
            self.stats['errors'] += 1
            return 0

    def anonymize_dataset(self, ds, keep_uids=False):
        ds_anon = ds.copy()  # Work on a copy
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
                        masked_value = tag_value_str  # Or just "REDACTED"
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
            ds_anon.ContentTime = now.strftime('%H%M%S.%f')[:14]  # Max 14 chars for TM VR

            # Ensure essential file_meta attributes are present and consistent
            if not hasattr(ds_anon, 'file_meta'):  # Should have been created if missing and redaction happened
                ds_anon.file_meta = pydicom.Dataset()

            # SOP Class UID - should exist in a valid DICOM
            if 'SOPClassUID' in ds_anon:
                 ds_anon.file_meta.MediaStorageSOPClassUID = ds_anon.SOPClassUID
            elif 'MediaStorageSOPClassUID' not in ds_anon.file_meta:  # if SOPClassUID also missing
                 logger.warning(f"SOPClassUID missing for {base_filename}, using default Secondary Capture. This may not be appropriate.")
                 ds_anon.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
                 if 'SOPClassUID' not in ds_anon: ds_anon.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage  # Also set in dataset

            # SOP Instance UID - ensure it matches if file_meta exists
            if 'SOPInstanceUID' in ds_anon:
                ds_anon.file_meta.MediaStorageSOPInstanceUID = ds_anon.SOPInstanceUID
            elif 'MediaStorageSOPInstanceUID' not in ds_anon.file_meta:  # if SOPInstanceUID also missing
                new_sop_instance_uid = generate_uid()  # Should have been generated if keep_uids=False
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
                ds_anon.file_meta.ImplementationVersionName = f"PYDICOM_ANON_LLM {pydicom.__version__}"

            try:
                # Ensure is_little_endian and is_implicit_VR match the TransferSyntaxUID
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
        self.stats = {k: 0 for k in self.stats}  # Reset stats for directory run
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

        for file_path in filter(Path.is_file, files_to_process):  # Ensure it's a file
            relative_path = file_path.relative_to(input_dir)
            output_file_path = output_dir / relative_path
            output_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output subdirectories exist
            self.anonymize_file(str(file_path), str(output_file_path), redact_overlays, keep_uids)

        logger.info("📊 Directory anonymization complete. Stats:")
        for k,v in self.stats.items():
            logger.info(f"  {k.replace('_',' ').capitalize()}: {v}")
        return self.stats


# Main execution block
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='DICOM Anonymizer with LLM-Based Text Sensitivity Detection')
    parser.add_argument('--input', required=True, help='Input DICOM file or directory')
    parser.add_argument('--output', required=True, help='Output DICOM file or directory')
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--pattern', default='*.dcm', help='File pattern (e.g., "*.dcm", "*.dicom")')
    parser.add_argument('--keep-uids', action='store_true', help='Keep original UIDs (Study, Series, SOP Instance)')
    parser.add_argument('--no-redact', action='store_true', help='Skip burned-in text redaction entirely')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'], help='Logging level')
    parser.add_argument('--confidence', type=float, default=0.4, help='OCR confidence threshold for redaction (0.0-1.0)')
    parser.add_argument('--ocr-gpu', action='store_true', help='Use GPU for EasyOCR if available.')
    parser.add_argument('--llm-model', default='gpt-4o-mini', help='OpenAI model to use for text classification')
    
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    anonymizer = DicomAnonymizer(
        ocr_gpu=args.ocr_gpu, 
        confidence_threshold=args.confidence,
        llm_model=args.llm_model
    )

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
           (output_path.exists() and output_path.is_dir()):  # More robust check if output_path is a directory
            output_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            output_file_path = output_path / input_path.name
        else:  # Assume output_path is a full file path
            output_file_path = output_path
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

        anonymizer.anonymize_file(str(input_path), str(output_file_path),
                                  redact_overlays=not args.no_redact,
                                  keep_uids=args.keep_uids)
    else:
        logger.error(f"Input path is not a valid file or directory: {input_path}")
        exit(1)

    logger.info("Anonymization process finished.")