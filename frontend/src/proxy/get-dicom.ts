"use server";

export const getDicom = async (result: any) => {
    try {
        const response = await fetch(`https://soun-backend.onrender.com/${result.file_url}`);
        const blob = await response.blob();
        const file = new File([blob], result.filename, {
            type: "application/dicom",
        });
        return file
    } catch (error) {
        console.error("Error loading anonymized file:", error);
        throw error;
    }
};
