import os
import sys
from pathlib import Path

# Import the DicomAnonymizer class from our improved anonymizer
from anonymizer import DicomAnonymizer

def get_dicom_file():
    """Prompt user for a valid DICOM file path."""
    path = input("🗂️ Enter the path to your DICOM file (.dcm): ").strip()
    while not os.path.exists(path) or not path.lower().endswith(".dcm"):
        print("⚠️ Invalid file. Please enter a valid .dcm file path.")
        path = input("🗂️ Enter the path to your DICOM file: ").strip()
    return path

def get_dicom_directory():
    """Prompt user for a valid directory containing DICOM files."""
    path = input("🗂️ Enter the directory containing DICOM files: ").strip()
    while not os.path.isdir(path):
        print("⚠️ Invalid directory. Please enter a valid directory path.")
        path = input("🗂️ Enter the directory containing DICOM files: ").strip()
    return path

def main():
    # Create an instance of our improved anonymizer
    anonymizer = DicomAnonymizer()
    
    while True:
        print("\n🔧 DICOM Privacy Tool")
        print("---------------------------")
        print("1️⃣ Anonymize Single DICOM File")
        print("2️⃣ Anonymize DICOM Directory")
        print("3️⃣ Advanced Anonymization Options")
        print("4️⃣ Exit")
        
        choice = input("Choose an option (1/2/3/4): ").strip()

        if choice == "1":
            # Anonymize a single file
            dicom_path = get_dicom_file()
            output_path = os.path.join(os.path.dirname(dicom_path), 
                                      "anonymized_" + os.path.basename(dicom_path))
            
            print(f"\n🔍 Processing: {dicom_path}")
            print(f"📤 Output will be saved to: {output_path}")
            
            # Force uncompressed output format by default to avoid compression errors
            success = anonymizer.anonymize_file(
                dicom_path, 
                output_path, 
               
            )
            
            if success:
                print(f"✅ Anonymization complete! File saved to: {output_path}")
            else:
                print(f"❌ Anonymization failed. Check log for details.")

        elif choice == "2":
            # Anonymize a directory of DICOM files
            input_dir = get_dicom_directory()
            output_dir = input_dir + "_anonymized"
            
            print(f"\n🔍 Processing directory: {input_dir}")
            print(f"📤 Output will be saved to: {output_dir}")
            
            recursive = input("📂 Process subdirectories? (y/n): ").strip().lower() == 'y'
            file_pattern = input("🔍 File pattern (default: *.dcm): ").strip() or "*.dcm"
            
            # Force uncompressed output format by default
            stats = anonymizer.anonymize_directory(
                input_dir, 
                output_dir, 
                recursive=recursive,
                file_pattern=file_pattern,
               
            
            )
            
            print("\n📊 Anonymization Statistics:")
            print(f"  ✓ Processed files: {stats['processed_files']}")
            print(f"  ✗ Skipped files: {stats['skipped_files']}")
            print(f"  🏷️ Removed tags: {stats['removed_tags']}")
            print(f"  🔍 Redacted text regions: {stats['redacted_text_regions']}")
            print(f"  ⚠️ Errors: {stats['errors']}")
            
            if stats['processed_files'] > 0:
                print(f"\n✅ Anonymization complete! Files saved to: {output_dir}")
            else:
                print(f"\n⚠️ No files were successfully processed.")

        elif choice == "3":
            # Advanced options submenu
            print("\n⚙️ Advanced Anonymization Options")
            print("---------------------------")
            
            dicom_path = get_dicom_file()
            output_path = os.path.join(os.path.dirname(dicom_path), 
                                      "anonymized_" + os.path.basename(dicom_path))
            
            # Get advanced options
            keep_uids = input("🔑 Keep original UIDs? (y/n): ").strip().lower() == 'y'
            redact_text = input("🔍 Redact burned-in text? (y/n): ").strip().lower() == 'y'
            keep_compressed = input("📦 Try to keep original compression? (y/n, default: n): ").strip().lower() == 'y'
            
            print(f"\n🔍 Processing: {dicom_path} with custom options")
            print(f"📤 Output will be saved to: {output_path}")
            
            success = anonymizer.anonymize_file(
                dicom_path, 
                output_path,
                redact_overlays=redact_text,
                keep_uids=keep_uids,
              
            )
            
            if success:
                print(f"✅ Advanced anonymization complete! File saved to: {output_path}")
            else:
                print(f"❌ Anonymization failed. Check log for details.")

        elif choice == "4":
            print("👋 Exiting. Stay private!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()