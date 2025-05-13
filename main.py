import os

from anonymizer import anonymize_dicom
from dicom_faker import insert_fake_data  # assuming you saved it in another file


def get_dicom_file():
    path = input("🗂️ Enter the path to your DICOM file (.dcm): ").strip()
    while not os.path.exists(path) or not path.lower().endswith(".dcm"):
        print("⚠️ Invalid file. Please enter a valid .dcm file path.")
        path = input("🗂️ Enter the path to your DICOM file: ").strip()
    return path

def main():
    while True:
        print("\n🔧 DICOM Privacy Tool")
        print("---------------------------")
        print("1️⃣ Insert Fake Data into DICOM (Unanonymize)")
        print("2️⃣ Anonymize DICOM")
        print("3️⃣ Exit")
        choice = input("Choose an option (1/2/3): ").strip()

        if choice == "1":
            dicom_path = get_dicom_file()
            output_path = "with_fake_data_" + os.path.basename(dicom_path)
            insert_fake_data(dicom_path, output_path)

        elif choice == "2":
            dicom_path = get_dicom_file()
            output_path = "anonymized_" + os.path.basename(dicom_path)
            anonymize_dicom(dicom_path, output_path)

        elif choice == "3":
            print("👋 Exiting. Stay private!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
