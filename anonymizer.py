import pydicom
from pydicom.uid import generate_uid

# Basic list of PHI tags to remove
SENSITIVE_TAGS = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0040),  # PatientSex
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x1010),  # StationName
    (0x0008, 0x1030),  # StudyDescription
    (0x0008, 0x1048),  # Physician(s) of Record
    (0x0008, 0x0050),  # AccessionNumber
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0010, 0x2160),  # EthnicGroup
    (0x0010, 0x21B0),  # AdditionalPatientHistory
    (0x0018, 0x1030),  # ProtocolName
    (0x0008, 0x1070),  # Operators' Name
    (0x0008, 0x009C),  # Consulting Physician
]

def anonymize_dicom(input_path, output_path):
    ds = pydicom.dcmread(input_path)

    print(f"\n📄 Processing: {input_path}")
    
    # Remove PHI tags
    for tag in SENSITIVE_TAGS:
        if tag in ds:
            print(f"❌ Removing {ds[tag].name}: {ds[tag].value}")
            del ds[tag]

    # Remove private vendor tags
    ds.remove_private_tags()

    # Replace UIDs with new ones
    if 'StudyInstanceUID' in ds:
        ds.StudyInstanceUID = generate_uid()
    if 'SeriesInstanceUID' in ds:
        ds.SeriesInstanceUID = generate_uid()
    if 'SOPInstanceUID' in ds:
        ds.SOPInstanceUID = generate_uid()

    # Clear burned-in text indicator
    if 'BurnedInAnnotation' in ds:
        ds.BurnedInAnnotation = 'NO'

    # Save anonymized output
    ds.save_as(output_path)
    print(f"✅ Anonymized file saved as: {output_path}")
