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
-docker

### Start the server
```bash
docker-compose up -d --build
```