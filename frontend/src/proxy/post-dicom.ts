'use server'

export const postDicom = async (file: any) => {
    try {
        const formData = new FormData();
        formData.append("file", file);

        // Add query parameters from the backend API definition
        const queryParams = new URLSearchParams({
            use_advanced: "false",
            redact_overlays: "true",
            keep_uids: "false",
            force_uncompressed: "true",
        });

        const url = `https://soun-backend.onrender.com/anonymize/?${queryParams}`;

        const response = await fetch(url, {
            method: "POST",
            body: formData,
            // Don't set Accept header to get the binary response
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }

        // Get the binary data
        const binaryData = await response.arrayBuffer();

        return binaryData;
    } catch (error) {
        console.error("Error during anonymization:", error);
        throw error;
    }
};
