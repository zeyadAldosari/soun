'use client';

import React, { useEffect, useRef, useState } from 'react';
import { UploadIcon, XIcon, Download } from 'lucide-react';
import { postDicom } from '@/proxy/post-dicom';
import Image from 'next/image';
import glowBlue from '@/assets/glow-blue-full.svg';
import glowPurple from '@/assets/glow-purple-full.svg';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';
import file1Image from '@/assets/file1.png';
import file2Image from '@/assets/file2.png';

// Remove the import for getDicom since we no longer need it
// import { getDicom } from "@/proxy/get-dicom";

// Don't import cornerstone libraries at the top level
// We'll import them dynamically in useEffect

const DicomViewer = () => {
  const [dicomFile, setDicomFile] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [cornerstoneLoaded, setCornerstoneLoaded] = useState(false);
  const [isBrowser, setIsBrowser] = useState(false);
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  const [isAnonymizedImageLoaded, setIsAnonymizedImageLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [processedResult, setProcessedResult] = useState(null);
  const [anonymizedDicomFile, setAnonymizedDicomFile] = useState(null);
  const [anonymizedMetadata, setAnonymizedMetadata] = useState(null);
  const imageRef = useRef(null);
  const anonymizedImageRef = useRef(null);

  // Effect to handle cleanup on unmount
  useEffect(() => {
    return () => {
      // Clean up cornerstone elements when component unmounts
      if (cornerstoneLoaded) {
        if (imageRef.current) {
          try {
            window.cornerstone.disable(imageRef.current);
          } catch (error) {
            console.error('Error disabling cornerstone:', error);
          }
        }
        if (anonymizedImageRef.current) {
          try {
            window.cornerstone.disable(anonymizedImageRef.current);
          } catch (error) {
            console.error(
              'Error disabling cornerstone for anonymized image:',
              error
            );
          }
        }
      }
    };
  }, [cornerstoneLoaded]);

  // Set isBrowser flag
  useEffect(() => {
    setIsBrowser(true);
  }, []);

  useEffect(() => {
    // Only proceed if we're in the browser
    if (!isBrowser) return;

    let cornerstone;
    let cornerstoneWADOImageLoader;
    let dicomParser;

    const loadLibraries = async () => {
      try {
        // Dynamically import the libraries
        cornerstone = (await import('cornerstone-core')).default;
        cornerstoneWADOImageLoader = (
          await import('cornerstone-wado-image-loader')
        ).default;
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

  // Effect to load and display the image when file is set
  useEffect(() => {
    // Only proceed if we have a file and cornerstone is loaded
    if (!dicomFile || !cornerstoneLoaded || !imageRef.current) return;

    const loadImage = async () => {
      try {
        const { cornerstone, cornerstoneWADOImageLoader } = window;

        // Enable cornerstone for the element
        cornerstone.enable(imageRef.current);

        // Create a file URL for the DICOM image
        const imageId =
          cornerstoneWADOImageLoader.wadouri.fileManager.add(dicomFile);

        // Load and display the image
        const image = await cornerstone.loadImage(imageId);
        cornerstone.displayImage(imageRef.current, image);

        // Add window width/center behavior
        imageRef.current.tabIndex = 0;
        imageRef.current.focus();
        cornerstone.resize(imageRef.current);

        setIsImageLoaded(true);
      } catch (error) {
        console.error('Error loading DICOM image:', error);
      }
    };

    loadImage();

    // Handle window resize for responsive behavior
    const handleResize = () => {
      if (imageRef.current && window.cornerstone) {
        try {
          window.cornerstone.resize(imageRef.current);
        } catch (error) {
          console.error('Error resizing cornerstone element:', error);
        }
      }
    };

    window.addEventListener('resize', handleResize);

    // Clean up when dicomFile changes or component unmounts
    return () => {
      window.removeEventListener('resize', handleResize);
      if (imageRef.current) {
        try {
          window.cornerstone.disable(imageRef.current);
          setIsImageLoaded(false);
        } catch (error) {
          console.error('Error disabling cornerstone:', error);
        }
      }
    };
  }, [dicomFile, cornerstoneLoaded]);

  // Add a new effect to load the anonymized image
  useEffect(() => {
    // Only proceed if we have an anonymized file and cornerstone is loaded
    if (
      !anonymizedDicomFile ||
      !cornerstoneLoaded ||
      !anonymizedImageRef.current
    )
      return;

    const loadAnonymizedImage = async () => {
      try {
        const { cornerstone, cornerstoneWADOImageLoader } = window;

        // Enable cornerstone for the element
        cornerstone.enable(anonymizedImageRef.current);

        // Create a file URL for the anonymized DICOM image
        const imageId =
          cornerstoneWADOImageLoader.wadouri.fileManager.add(
            anonymizedDicomFile
          );

        // Load and display the image
        const image = await cornerstone.loadImage(imageId);
        cornerstone.displayImage(anonymizedImageRef.current, image);

        // Add window width/center behavior
        anonymizedImageRef.current.tabIndex = 0;
        anonymizedImageRef.current.focus();
        cornerstone.resize(anonymizedImageRef.current);

        setIsAnonymizedImageLoaded(true);
      } catch (error) {
        console.error('Error loading anonymized DICOM image:', error);
      }
    };

    loadAnonymizedImage();

    // Handle window resize for responsive behavior
    const handleResize = () => {
      if (anonymizedImageRef.current && window.cornerstone) {
        try {
          window.cornerstone.resize(anonymizedImageRef.current);
        } catch (error) {
          console.error(
            'Error resizing cornerstone element for anonymized image:',
            error
          );
        }
      }
    };

    window.addEventListener('resize', handleResize);

    // Clean up when anonymizedDicomFile changes or component unmounts
    return () => {
      window.removeEventListener('resize', handleResize);
      if (anonymizedImageRef.current) {
        try {
          window.cornerstone.disable(anonymizedImageRef.current);
          setIsAnonymizedImageLoaded(false);
        } catch (error) {
          console.error(
            'Error disabling cornerstone for anonymized image:',
            error
          );
        }
      }
    };
  }, [anonymizedDicomFile, cornerstoneLoaded]);

  const handleFileChange = async (event) => {
    if (
      !cornerstoneLoaded ||
      !event.target.files ||
      event.target.files.length === 0
    )
      return;

    const file = event.target.files[0];

    try {
      const { dicomParser } = window;
      const arrayBuffer = await file.arrayBuffer();
      const dataSet = dicomParser.parseDicom(new Uint8Array(arrayBuffer));

      const extractedMetadata = {
        patientID: dataSet.string('x00100020') || 'Unknown',
        patientName: dataSet.string('x00100010') || 'Unknown',
        modality: dataSet.string('x00080060') || 'Unknown',
        seriesInstanceUID: dataSet.string('x0020000e') || 'Unknown',
        studyInstanceUID: dataSet.string('x0020000d') || 'Unknown',
        studyDate: dataSet.string('x00080020') || 'Unknown',
      };

      setMetadata(extractedMetadata);
      // Set dicomFile after metadata is successfully extracted
      setDicomFile(file);
      // Clear any previous results
      setProcessedResult(null);
      setAnonymizedDicomFile(null);
      setAnonymizedMetadata(null);
      setIsAnonymizedImageLoaded(false);
    } catch (error) {
      console.error('Error parsing DICOM file:', error);
    }
  };

  const handleFileChangeDefault = async (fileName) => {
    try {
      const response = await fetch(fileName);
      const blob = await response.blob();
      const file = new File([blob], 'file.dcm');

      const { dicomParser } = window;
      const arrayBuffer = await file.arrayBuffer();
      const dataSet = dicomParser.parseDicom(new Uint8Array(arrayBuffer));

      const extractedMetadata = {
        patientID: dataSet.string('x00100020') || 'Unknown',
        patientName: dataSet.string('x00100010') || 'Unknown',
        modality: dataSet.string('x00080060') || 'Unknown',
        seriesInstanceUID: dataSet.string('x0020000e') || 'Unknown',
        studyInstanceUID: dataSet.string('x0020000d') || 'Unknown',
        studyDate: dataSet.string('x00080020') || 'Unknown',
      };

      setMetadata(extractedMetadata);
      // Set dicomFile after metadata is successfully extracted
      setDicomFile(file);
      // Clear any previous results
      setProcessedResult(null);
      setAnonymizedDicomFile(null);
      setAnonymizedMetadata(null);
      setIsAnonymizedImageLoaded(false);
    } catch (error) {
      console.error('Error parsing DICOM file:', error);
    }
  };

  const clearFile = () => {
    // Clean up cornerstone if viewing
    if (cornerstoneLoaded) {
      if (imageRef.current) {
        try {
          window.cornerstone.disable(imageRef.current);
        } catch (error) {
          console.error('Error disabling cornerstone:', error);
        }
      }
      if (anonymizedImageRef.current) {
        try {
          window.cornerstone.disable(anonymizedImageRef.current);
        } catch (error) {
          console.error(
            'Error disabling cornerstone for anonymized image:',
            error
          );
        }
      }
    }

    setDicomFile(null);
    setMetadata(null);
    setIsImageLoaded(false);
    setProcessedResult(null);
    setAnonymizedDicomFile(null);
    setAnonymizedMetadata(null);
    setIsAnonymizedImageLoaded(false);

    // Reset file input
    const fileInput = document.getElementById('dicom-file-input');
    if (fileInput) fileInput.value = '';
  };

  const handleAnonymize = async () => {
    setIsLoading(true);
    try {
      // The postDicom function now returns binary data instead of a file URL
      const binaryData = await postDicom(dicomFile);

      // Create a file from the binary data
      const filename = dicomFile.name.replace('.dcm', '_anonymized.dcm');
      const file = new File([binaryData], filename, {
        type: 'application/dicom',
      });

      // Set the result object with basic information
      const result = {
        filename: filename,
        message: 'تمت عملية إخفاء الهوية بنجاح',
      };

      setProcessedResult(result);
      console.log('Anonymization completed:', result);

      try {
        // Parse the anonymized DICOM file to extract metadata
        const { dicomParser } = window;
        const arrayBuffer = await file.arrayBuffer();
        const dataSet = dicomParser.parseDicom(new Uint8Array(arrayBuffer));

        const extractedMetadata = {
          patientID: dataSet.string('x00100020') || 'Anonymous',
          patientName: dataSet.string('x00100010') || 'Anonymous',
          modality: dataSet.string('x00080060') || 'Unknown',
          seriesInstanceUID: dataSet.string('x0020000e') || 'Unknown',
          studyInstanceUID: dataSet.string('x0020000d') || 'Unknown',
          studyDate: dataSet.string('x00080020') || 'Unknown',
        };

        setAnonymizedMetadata(extractedMetadata);
        setAnonymizedDicomFile(file);
      } catch (error) {
        console.error('Error parsing anonymized DICOM file:', error);
      }
    } catch (e) {
      console.error('Error hiding identity:', e);
      alert('حدث خطأ أثناء إخفاء الهوية.');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to create a download URL for the anonymized file
  const createDownloadUrl = () => {
    if (!anonymizedDicomFile) return null;
    return URL.createObjectURL(anonymizedDicomFile);
  };

  return (
    <div className="w-full max-w-3xl mx-auto gap-4 px-4 sm:px-6 ">
      <div className="w-full relative gap-4 flex flex-col">
        <Image
          src={glowBlue}
          alt="glow"
          className="absolute w-96 aspect-square -right-56 opacity-50"
        />
        <Image
          src={glowPurple}
          alt="glow"
          className="absolute w-72 aspect-square -right-10 top-0 opacity-30"
        />
        {!dicomFile && (
          <>
            <div className="p-2 grid place-items-center relative bg-[#404040]/9 border border-[#404040]/10 rounded-2xl bg-opacity-20 backdrop-blur-sm w-full h-64 sm:h-80 md:h-96 group">
              <div className="w-full h-full border-2 border-[#303030] border-dashed rounded-2xl grid place-items-center text-[#303030] group-hover:border-[#4a4a4a] transition-colors">
                <input
                  id="dicom-file-input"
                  className="w-full h-full absolute opacity-0 cursor-pointer peer"
                  type="file"
                  accept=".dcm"
                  onChange={handleFileChange}
                  disabled={!cornerstoneLoaded}
                />
                <div className="grid place-items-center text-lg sm:text-xl md:text-2xl font-bold gap-2 peer-hover:text-[#4a4a4a] transition-colors px-4 text-center">
                  <UploadIcon className="w-12 h-12 sm:w-16 sm:h-16 md:w-20 md:h-20 lg:w-24 lg:h-24" />
                  ارفق الملف الذي تريد تحويله
                </div>
              </div>
            </div>
            <div className="w-full text-white font-bold">
              <p>او جرب احد الملفات التالية:</p>
              <div className="flex justify-center gap-4 mt-4">
                <button
                  onClick={() => {
                    handleFileChangeDefault('file1fake.dcm');
                  }}
                  className="bg-[#404040]/9 border border-[#404040]/10 rounded-2xl bg-opacity-20 backdrop-blur-sm hover:bg-[hsl(0,0%,15%)] text-white transition-colors text-sm sm:text-base cursor-pointer w-32 aspect-square relative overflow-hidden"
                >
                  <Image
                    src={file1Image}
                    fill
                    sizes="100%"
                    alt="image"
                    className="absolute opacity-20"
                  />
                  <span className="z-10">الملف 1</span>
                </button>
                <button
                  onClick={() => {
                    handleFileChangeDefault('file2fake.dcm');
                  }}
                  className="bg-[#404040]/9 border border-[#404040]/10 rounded-2xl bg-opacity-20 backdrop-blur-sm hover:bg-[hsl(0,0%,15%)] text-white transition-colors text-sm sm:text-base cursor-pointer w-32 aspect-square overflow-hidden"
                >
                  <Image
                    src={file2Image}
                    fill
                    sizes="100%"
                    alt="image"
                    className="absolute opacity-20"
                  />
                  <span className="z-10">الملف 2</span>
                </button>
              </div>
            </div>
          </>
        )}

        {/* Loading message */}
        {isBrowser && !cornerstoneLoaded && (
          <div className="p-4 bg-[#404040]/9 border border-[#404040]/10 rounded-2xl backdrop-blur-sm text-[#303030] text-center w-full">
            <DotLottieReact src="loading.lottie" loop autoplay />
          </div>
        )}

        {/* File details and image display combined section */}
        {dicomFile && metadata && (
          <div className="p-4 sm:p-6 bg-[#404040]/9 border border-[#404040]/10 rounded-2xl backdrop-blur-sm w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg sm:text-xl font-bold text-white">
                تفاصيل الملف
              </h2>
              <button
                onClick={clearFile}
                className="p-2 rounded-full hover:bg-[#404040]/20 transition-colors cursor-pointer"
                aria-label="Clear file"
              >
                <XIcon size={20} className="text-[#303030]" />
              </button>
            </div>

            {/* Metadata display - responsive grid */}
            <div className="grid w-full grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-4 text-white mb-6 text-sm sm:text-base">
              <div className="p-2 border-b border-[#404040]/20 overflow-hidden text-ellipsis">
                <span className="font-semibold text-white">اسم الملف: </span>
                <span className="ml-2 break-words">{dicomFile.name}</span>
              </div>
              <div className="p-2 border-b border-[#404040]/20">
                <span className="font-semibold">معرف المريض: </span>
                <span className="ml-2 break-words">{metadata.patientID}</span>
              </div>
              <div className="p-2 border-b border-[#404040]/20">
                <span className="font-semibold">اسم المريض: </span>
                <span className="ml-2 break-words">{metadata.patientName}</span>
              </div>
              <div className="p-2 border-b border-[#404040]/20">
                <span className="font-semibold">النمط: </span>
                <span className="ml-2">{metadata.modality}</span>
              </div>
            </div>

            {/* DICOM Image Display */}
            <h3 className="text-base sm:text-lg font-semibold text-white mb-2">
              الصورة
            </h3>
            <div
              className="w-full rounded-xl overflow-hidden bg-black"
              ref={imageRef}
              style={{
                height: '300px',
                maxHeight: '70vh',
              }}
            />

            {/* Loading indicator for image */}
            {!isImageLoaded && (
              <div className="w-full">
                <DotLottieReact
                  className="w-32 aspect-square"
                  src="loading.lottie"
                  loop
                  autoplay
                />
              </div>
            )}

            {/* Anonymize button */}
            <div className="flex justify-center sm:justify-start mt-4">
              <button
                onClick={handleAnonymize}
                disabled={isLoading}
                className="w-full sm:w-auto px-4 sm:px-6 py-2 sm:py-3 bg-[#303030] hover:bg-[#4a4a4a] text-white rounded-xl transition-colors text-sm sm:text-base cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                إخفاء الهوية
              </button>
            </div>
            {isLoading && (
              <div className="w-full grid place-items-center ">
                <DotLottieReact
                  className="w-32 aspect-square"
                  src="loading.lottie"
                  loop
                  autoplay
                />
              </div>
            )}
          </div>
        )}

        {/* Results Card - shown after processing */}
        {processedResult && (
          <div className="p-4 sm:p-6 bg-[#404040]/9 border border-[#404040]/10 rounded-2xl backdrop-blur-sm w-full text-white">
            <Image
              src={glowBlue}
              alt="glow"
              className="absolute w-96 aspect-square -left-56 opacity-50"
            />
            <Image
              src={glowPurple}
              alt="glow"
              className="absolute w-72 aspect-square -left-10 top-0 opacity-30"
            />
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg sm:text-xl font-bold text-white">
                نتيجة إخفاء الهوية
              </h2>
            </div>

            {/* Result display - responsive grid */}
            <div className="grid w-full grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-4 text-white mb-6 text-sm sm:text-base">
              <div className="p-2 border-b border-[#404040]/20 overflow-hidden text-ellipsis">
                <span className="font-semibold">اسم الملف الجديد: </span>
                <span className="ml-2 break-words">
                  {processedResult.filename}
                </span>
              </div>
              <div className="p-2 border-b border-[#404040]/20">
                <span className="font-semibold">الحالة: </span>
                <span className="ml-2 break-words">
                  {processedResult.message}
                </span>
              </div>
            </div>

            {/* Anonymized Metadata display if available */}
            {anonymizedMetadata && (
              <>
                <h3 className="text-base sm:text-lg font-semibold text-white mb-2">
                  بيانات الملف المجهول الهوية
                </h3>
                <div className="grid w-full grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-4 text-white mb-6 text-sm sm:text-base">
                  <div className="p-2 border-b border-[#404040]/20">
                    <span className="font-semibold">معرف المريض: </span>
                    <span className="ml-2 break-words">
                      {anonymizedMetadata.patientID}
                    </span>
                  </div>
                  <div className="p-2 border-b border-[#404040]/20">
                    <span className="font-semibold">اسم المريض: </span>
                    <span className="ml-2 break-words">
                      {anonymizedMetadata.patientName}
                    </span>
                  </div>
                  <div className="p-2 border-b border-[#404040]/20">
                    <span className="font-semibold">النمط: </span>
                    <span className="ml-2">{anonymizedMetadata.modality}</span>
                  </div>
                </div>
              </>
            )}

            {/* Anonymized DICOM Image Display */}
            {anonymizedDicomFile && (
              <>
                <h3 className="text-base sm:text-lg font-semibold text-white mb-2">
                  الصورة المجهولة الهوية
                </h3>
                <div
                  className="w-full rounded-xl overflow-hidden bg-black"
                  ref={anonymizedImageRef}
                  style={{
                    height: '300px',
                    maxHeight: '70vh',
                  }}
                />

                {/* Loading indicator for anonymized image */}
                {!isAnonymizedImageLoaded && (
                  <div className="h-20 aspect-square">
                    <DotLottieReact src="loading.lottie" loop autoplay />
                  </div>
                )}
              </>
            )}

            {/* Download button - Now using a local blob URL instead of the server URL */}
            {anonymizedDicomFile && (
              <div className="flex justify-center mt-4">
                <a
                  href={createDownloadUrl()}
                  download={processedResult.filename}
                  className="flex items-center gap-2 px-4 sm:px-6 py-2 sm:py-3 bg-[#303030] hover:bg-[#4a4a4a] text-white rounded-xl transition-colors text-sm sm:text-base cursor-pointer"
                >
                  <Download size={18} />
                  تنزيل الملف المجهول الهوية
                </a>
              </div>
            )}
          </div>
        )}
      </div>

      {/* File upload area - shown when no file is selected */}
    </div>
  );
};

export default React.memo(DicomViewer);
