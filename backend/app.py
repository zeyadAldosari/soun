from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
import os
import shutil
import uuid
import zipfile
from pydantic import BaseModel
import uvicorn
import logging
from typing import Optional
import time
from utils.anonymizer import DicomAnonymizer
from utils.dicom_faker import insert_fake_data
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi import Header
import json


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI(
    title="DICOM Privacy Tool API",
    description="API for anonymizing DICOM files or inserting fake data",
    version="1.1.0"
)

UPLOAD_DIR = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

class ProcessResponse(BaseModel):
    """Response model for DICOM processing endpoints"""
    filename: str
    message: str
    file_url: str    
    stats: Optional[dict] = None


@app.post("/insert-fake-data/", response_model=ProcessResponse)
async def insert_fake_data_endpoint(file: UploadFile = File(...)):
    """
    Insert fake data into a DICOM file (.dcm)
    
    This endpoint will create a new DICOM file with synthetic patient data
    """
    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(status_code=400, detail="Only .dcm files are accepted")
    
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_filename = f"with_fake_data_{file.filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    
    try:
        insert_fake_data(temp_file_path, output_path)
        return ProcessResponse(
            filename=output_filename,
            message="Fake data inserted successfully"
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    

@app.post("/anonymize/")
async def anonymize_endpoint(
    file: UploadFile = File(...),
    use_advanced: bool = Query(False, description="Use the advanced anonymizer implementation"),
    redact_overlays: bool = Query(True, description="Attempt to redact burned-in text"),
    keep_uids: bool = Query(False, description="Keep original UIDs"),
    force_uncompressed: bool = Query(True, description="Convert to uncompressed transfer syntax"),
    accept: str = Header(None)  # Get the Accept header to determine response format
):
    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(status_code=400, detail="Only .dcm files are accepted")
   
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
       
    timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_filename = f"anonymized_{timestamp_str}_{file.filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
   
    try:
        stats = None
        if use_advanced:
            logger.info(f"Using advanced anonymizer for {file.filename}")
            anonymizer = DicomAnonymizer(ocr_languages=['en'], ocr_gpu=False)
            success = anonymizer.anonymize_file(
                temp_file_path,
                output_path,
                redact_overlays=redact_overlays,
                keep_uids=keep_uids,
                force_uncompressed=force_uncompressed
            )
            stats = anonymizer.stats
            if not success:
                raise Exception("Advanced anonymization failed")
        else:
            logger.info(f"Using original anonymizer for {file.filename}")
            anonymizer = DicomAnonymizer()
            anonymizer.anonymize_file(temp_file_path, output_path)
       
        file_url = f"/files/{output_filename}"
        
        # Create metadata dictionary
        metadata = {
            "filename": output_filename,
            "message": "DICOM file anonymized successfully",
            "file_url": file_url,
            "stats": stats
        }
        
        # If client requests JSON, return just the metadata with file_url
        if accept and "application/json" in accept:
            return metadata
            
        # Otherwise stream the binary file with metadata in headers
        def iterfile():
            with open(output_path, mode="rb") as file_like:
                yield from file_like
        
        # Create a StreamingResponse with the DICOM file data
        response = StreamingResponse(
            iterfile(),
            media_type="application/dicom"
        )
        
        # Add metadata as headers
        response.headers["X-Filename"] = output_filename
        response.headers["X-Message"] = "DICOM file anonymized successfully"
        response.headers["X-File-URL"] = file_url
        
        # If stats exist, convert to JSON string and add as header
        if stats:
            response.headers["X-Stats"] = json.dumps(stats)
            
        return response
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a processed DICOM file
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type="application/dicom")

@app.post("/anonymize-batch/", response_model=ProcessResponse)
async def anonymize_batch_endpoint(
    file: UploadFile = File(...),
    use_advanced: bool = Query(False, description="Use the advanced anonymizer implementation"),
    redact_overlays: bool = Query(True, description="Attempt to redact burned-in text"),
    keep_uids: bool = Query(False, description="Keep original UIDs"),
    force_uncompressed: bool = Query(True, description="Convert to uncompressed transfer syntax"),
    recursive: bool = Query(True, description="Process subdirectories in the zip file")
):
    """
    Anonymize multiple DICOM files from a zip archive
    
    This endpoint accepts a zip file containing DICOM files (.dcm),
    extracts them, anonymizes each file, and returns a zip file with the results.
    """
    # Validate file is a zip
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")
    
    # Create unique session ID for this batch
    session_id = str(uuid.uuid4())
    
    # Create temporary directories for input and output files
    temp_input_dir = os.path.join(UPLOAD_DIR, f"input_{session_id}")
    temp_output_dir = os.path.join(UPLOAD_DIR, f"output_{session_id}")
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Save the uploaded zip file
    zip_path = os.path.join(UPLOAD_DIR, f"upload_{session_id}.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_input_dir)
        
        # Anonymize all DICOM files in the directory
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        if use_advanced:
            logger.info(f"Using advanced anonymizer for batch processing")
            anonymizer = DicomAnonymizer(ocr_languages=['en'], ocr_gpu=False)
            stats = anonymizer.anonymize_directory(
                temp_input_dir,
                temp_output_dir,
                recursive=recursive,
                file_pattern='*.dcm',
                redact_overlays=redact_overlays,
                keep_uids=keep_uids
            )
        else:
            logger.info(f"Using original anonymizer for batch processing")
            anonymizer = DicomAnonymizer()
            stats = anonymizer.anonymize_directory(
                temp_input_dir,
                temp_output_dir,
                recursive=recursive,
                file_pattern='*.dcm',
                redact_overlays=redact_overlays,
                keep_uids=keep_uids
            )
        output_zip_filename = f"anonymized_batch_{timestamp_str}.zip"
        output_zip_path = os.path.join(UPLOAD_DIR, output_zip_filename)
        
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_output_dir)
                    zipf.write(file_path, arcname)
        
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        os.remove(zip_path)
        
        return ProcessResponse(
            filename=output_zip_filename,
            message=f"Batch anonymization complete. {stats['processed_files']} files processed.",
            stats=stats
        )
        
    except Exception as e:
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/")
def read_root():
    """
    API welcome page with endpoint information
    """
    return {
        "message": "Welcome to DICOM Privacy Tool API",
        "endpoints": {
            "POST /insert-fake-data/": "Insert fake data into a DICOM file",
            "POST /anonymize/": "Anonymize a DICOM file (supports original or advanced implementation)",
            "GET /download/{filename}": "Download a processed file"
        }
    }


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Remove files older than the specified age from a directory
    """
    current_time = time.time()
    max_age_seconds = max_age_hours * 60 * 60
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_age = current_time - os.path.getmtime(file_path)
        
        if file_age > max_age_seconds:
            try:
                os.remove(file_path)
                logger.info(f"Removed old file: {filename}")
            except Exception as e:
                logger.error(f"Error removing {filename}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run cleanup on startup to remove old files
    """
    cleanup_old_files(UPLOAD_DIR)
    yield


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)