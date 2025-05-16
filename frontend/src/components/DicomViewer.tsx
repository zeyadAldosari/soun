'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  Modal,
  Box,
  Typography,
} from '@mui/material';

// Don't import cornerstone libraries at the top level
// We'll import them dynamically in useEffect

interface ImageMetadata {
  patientID: string;
  modality: string;
  seriesInstanceUID: string;
  studyInstanceUID: string;
  files: File[];
}

const DicomViewer = () => {
  const [metadata, setMetadata] = useState<ImageMetadata[]>([]);
  const [open, setOpen] = useState<boolean>(false);
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const imageRefs = useRef<HTMLDivElement[]>([]);
  const [cornerstoneLoaded, setCornerstoneLoaded] = useState(false);
  // Add state to track if we're in browser environment
  const [isBrowser, setIsBrowser] = useState(false);
  
  // First useEffect just to set isBrowser
  useEffect(() => {
    setIsBrowser(true);
  }, []);
  
  // Dynamic imports for client-side only libraries
  useEffect(() => {
    // Only proceed if we're in the browser
    if (!isBrowser) return;
    
    let cornerstone: any;
    let cornerstoneWADOImageLoader: any;
    let dicomParser: any;
    
    const loadLibraries = async () => {
      try {
        // Dynamically import the libraries
        cornerstone = (await import('cornerstone-core')).default;
        cornerstoneWADOImageLoader = (await import('cornerstone-wado-image-loader')).default;
        dicomParser = (await import('dicom-parser')).default;
        
        // Configure the libraries
        cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
        cornerstoneWADOImageLoader.external.dicomParser = dicomParser;
        cornerstoneWADOImageLoader.configure({});
        
        // Store them on window for later access
        window.cornerstone = cornerstone;
        window.cornerstoneWADOImageLoader = cornerstoneWADOImageLoader;
        window.dicomParser = dicomParser;
        
        setCornerstoneLoaded(true);
      } catch (error) {
        console.error('Error loading cornerstone libraries:', error);
      }
    };

    loadLibraries();
  }, [isBrowser]);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!cornerstoneLoaded || !event.target.files) return;
    
    const filesArray = Array.from(event.target.files);
    const { dicomParser } = window as any;

    for (const file of filesArray) {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const dataSet = dicomParser.parseDicom(new Uint8Array(arrayBuffer));
        
        const patientID = dataSet.string('x00100020') || 'Unknown';
        const modality = dataSet.string('x00080060') || 'Unknown';
        const seriesInstanceUID = dataSet.string('x0020000e') || 'Unknown';
        const studyInstanceUID = dataSet.string('x0020000d') || 'Unknown';

        setMetadata((prevMetadata) => {
          const existingPatient = prevMetadata.find(
            (data) => data.patientID === patientID,
          );
          if (existingPatient) {
            existingPatient.files.push(file);
            return [...prevMetadata];
          } else {
            return [
              ...prevMetadata,
              {
                patientID,
                modality,
                seriesInstanceUID,
                studyInstanceUID,
                files: [file],
              },
            ];
          }
        });
      } catch (error) {
        console.error('Error parsing DICOM file:', error);
      }
    }
  };

  const handleOpenModal = (files: File[]) => {
    const uniqueFiles = Array.from(new Set(files.map((file) => file.name))).map(
      (uniqueFileName) => {
        return files.find((file) => file.name === uniqueFileName);
      },
    ) as File[];

    setSelectedImages(uniqueFiles);
    setOpen(true);
    setTimeout(() => {
      uniqueFiles.forEach((file, index) => {
        loadImage(file, index);
      });
    }, 0);
  };

  const handleCloseModal = useCallback(() => {
    setOpen(false);
    setSelectedImages([]);
  }, [setOpen, setSelectedImages]);

  const loadImage = async (file: File, index: number) => {
    if (!cornerstoneLoaded) return;
    
    const element = imageRefs.current[index];
    if (element) {
      const { cornerstone, cornerstoneWADOImageLoader } = window as any;
      
      cornerstone.enable(element);
      const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
      try {
        const image = await cornerstone.loadImage(imageId);
        cornerstone.displayImage(element, image);
      } catch (error) {
        console.error('Error loading DICOM image:', error);
      }
    }
  };

  return (
    <main>
      <div className="container">
        <input 
        className='border-2 border-[#303030] rounded-2xl w-32 h-32'
          type="file" 
          accept=".dcm" 
          multiple 
          onChange={handleFileChange}
          disabled={!cornerstoneLoaded}
        />
        {isBrowser && !cornerstoneLoaded && (
          <Typography color="error">Loading DICOM libraries...</Typography>
        )}
        {metadata.length > 0 && (
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Patient ID</TableCell>
                  <TableCell>Modality</TableCell>
                  <TableCell>Series Instance UID</TableCell>
                  <TableCell>Study Instance UID</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {metadata.map((data, index) => (
                  <TableRow key={index}>
                    <TableCell>{data.patientID}</TableCell>
                    <TableCell>{data.modality}</TableCell>
                    <TableCell>{data.seriesInstanceUID}</TableCell>
                    <TableCell>{data.studyInstanceUID}</TableCell>
                    <TableCell>
                      <Button
                        variant="contained"
                        onClick={() => handleOpenModal(data.files)}
                      >
                        Show Images
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </div>
      <Modal open={open} onClose={handleCloseModal}>
        <Box className="box">
          <Typography variant="h6" component="h2" textAlign="center">
            DICOM Images
          </Typography>
          <div className="image-container">
            {selectedImages.map((_, index) => (
              <div key={index}>
                <div
                  ref={(el) => {
                    if (el) imageRefs.current[index] = el;
                  }}
                  className="images-wrapper"
                />
              </div>
            ))}
          </div>
        </Box>
      </Modal>
    </main>
  );
};

// Adding TypeScript interface for window object
declare global {
  interface Window {
    cornerstone: any;
    cornerstoneWADOImageLoader: any;
    dicomParser: any;
  }
}

export default React.memo(DicomViewer);