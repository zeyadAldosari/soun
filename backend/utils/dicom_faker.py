from faker import Faker
import pydicom
import random
fake = Faker()

def insert_fake_data(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)

    # Insert fake data
    ds.PatientName = fake.name()
    ds.PatientID = str(fake.random_number(digits=8))
    ds.PatientBirthDate = fake.date_of_birth().strftime('%Y%m%d')
    ds.PatientSex = random.choice(['M', 'F'])
    ds.InstitutionName = fake.company()
    ds.ReferringPhysicianName = fake.name()
    ds.StationName = fake.word()
    ds.StudyDescription = fake.sentence(nb_words=4)
    ds.ProtocolName = fake.bs()
    ds.OperatorsName = fake.name()
    ds.AccessionNumber = str(fake.random_number(digits=6))
    ds.AdditionalPatientHistory = fake.text(max_nb_chars=50)

    ds.save_as(output_path)
    print(f"🧪 Fake data inserted and saved as: {output_path}")
