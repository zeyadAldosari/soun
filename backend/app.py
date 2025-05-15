import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import uvicorn
from pydantic import BaseModel
from utils.anonymizer import anonymize_dicom
from utils.dicom_faker import insert_fake_data

app = FastAPI(
    title="DICOM Privacy Tool API",
    description="API for anonymizing DICOM files or inserting fake data",
    version="1.0.0"
)
UPLOAD_DIR = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ProcessResponse(BaseModel):
    """Response model for DICOM processing endpoints"""
    filename: str
    message: str


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
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/anonymize/", response_model=ProcessResponse)
async def anonymize_endpoint(file: UploadFile = File(...)):
    """
    Anonymize a DICOM file (.dcm)
    
    This endpoint will create a new anonymized version of the input DICOM file
    """
    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(status_code=400, detail="Only .dcm files are accepted")
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    output_filename = f"anonymized_{file.filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    try:
        anonymize_dicom(temp_file_path, output_path)
        return ProcessResponse(
            filename=output_filename,
            message="DICOM file anonymized successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type="application/dicom")


@app.get("/")
def read_root():
    return {
        "message": "Welcome to DICOM Privacy Tool API",
        "endpoints": {
            "POST /insert-fake-data/": "Insert fake data into a DICOM file",
            "POST /anonymize/": "Anonymize a DICOM file",
            "GET /download/{filename}": "Download a processed file"
        }
    }

def cleanup_old_files(directory: str, max_age_hours: int = 24):
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 60 * 60
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_age = current_time - os.path.getmtime(file_path)
        
        if file_age > max_age_seconds:
            try:
                os.remove(file_path)
                print(f"Removed old file: {filename}")
            except Exception as e:
                print(f"Error removing {filename}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)