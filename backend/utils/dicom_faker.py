from faker import Faker
import pydicom
import datetime
import random

fake = Faker()

def insert_fake_data(input_path, output_path):
    """
    Insert fake patient data into a DICOM file
    
    Args:
        input_path: Path to input DICOM file
        output_path: Path to output DICOM file with fake data
    """
    ds = pydicom.dcmread(input_path)
    patient_name = fake.name()
    patient_id = fake.random_number(digits=8)
    patient_birth_date = fake.date_of_birth(minimum_age=18, maximum_age=90)
    patient_sex = random.choice(['M', 'F', 'O'])
    formatted_birth_date = patient_birth_date.strftime("%Y%m%d")
    ds.PatientName = patient_name
    ds.PatientID = str(patient_id)
    ds.PatientBirthDate = formatted_birth_date
    ds.PatientSex = patient_sex
    ds.InstitutionName = fake.company() + " Hospital"
    ds.ReferringPhysicianName = fake.name() + ", M.D."
    today = datetime.date.today()
    age = today.year - patient_birth_date.year
    if today.month < patient_birth_date.month or (today.month == patient_birth_date.month and today.day < patient_birth_date.day):
        age -= 1
    ds.PatientAge = f"{age:03d}Y"
    ds.save_as(output_path)
    print(f"✅ Created DICOM with fake data for patient: {patient_name}")
    
    return ds