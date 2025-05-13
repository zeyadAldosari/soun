# DICOM Remover Tool

## Overview
DICOM Remover is a specialized utility designed for healthcare professionals, researchers, and IT administrators who need to safely remove or anonymize DICOM (Digital Imaging and Communications in Medicine) files. This tool helps manage medical imaging data while ensuring compliance with privacy regulations such as HIPAA.

## Features
- **Selective DICOM Removal**: Remove specific DICOM files based on various criteria
- **Batch Processing**: Process multiple DICOM files or entire directories at once
- **Data Anonymization**: Remove or modify patient identifiable information from DICOM files
- **Metadata Preservation**: Maintain important medical metadata while removing sensitive information
- **Audit Logging**: Keep detailed logs of all removal and modification operations
- **Preview Mode**: Review changes before permanent application
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Install from PyPI
```bash
pip install dicom-remover
```

### Install from Source
```bash
git clone https://github.com/yourusername/dicom-remover.git
cd dicom-remover
pip install -e .
```

## Usage

### Basic Usage
```bash
# Remove all DICOM files in a directory
dicom-remover --input /path/to/dicom/files --output /path/to/output

# Anonymize DICOM files
dicom-remover --input /path/to/dicom/files --output /path/to/output --anonymize

# Preview changes without applying them
dicom-remover --input /path/to/dicom/files --dry-run
```

### Advanced Options
```bash
# Remove specific tags from DICOM files
dicom-remover --input /path/to/dicom/files --output /path/to/output --remove-tags PatientName,PatientID

# Apply custom anonymization rules
dicom-remover --input /path/to/dicom/files --output /path/to/output --config /path/to/config.json
```

## Configuration
Create a JSON configuration file to customize the tool's behavior:

```json
{
  "remove_tags": ["PatientName", "PatientID", "PatientBirthDate"],
  "replace_tags": {
    "PatientName": "ANONYMOUS",
    "InstitutionName": "RESEARCH FACILITY"
  },
  "keep_tags": ["StudyDescription", "SeriesDescription"],
  "log_level": "info"
}
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- [pydicom](https://github.com/pydicom/pydicom) - Python package for working with DICOM files
- DICOM Standard - [NEMA](https://www.dicomstandard.org/)