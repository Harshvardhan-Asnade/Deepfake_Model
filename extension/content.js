/**
 * DeepGuard Browser Extension - Content Script
 * Handles image capture and API communication from web pages
 */

const API_URL = 'http://localhost:5001/api/predict';

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'analyzeImage') {
        analyzeImageFromUrl(request.imageUrl)
            .then(result => {
                sendResponse({ success: true, result });
            })
            .catch(error => {
                console.error('Error analyzing image:', error);
                sendResponse({ success: false, error: error.message });
            });

        return true; // Keep channel open for async response
    }
});

/**
 * Fetch image from URL and send to API for analysis
 */
async function analyzeImageFromUrl(imageUrl) {
    try {
        // Fetch the image
        const response = await fetch(imageUrl);
        const blob = await response.blob();

        // Create FormData
        const formData = new FormData();
        formData.append('file', blob, 'image.png');

        // Send to API
        const apiResponse = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        const data = await apiResponse.json();

        if (!data.error) {
            return data;
        } else {
            throw new Error(data.error || 'API request failed');
        }
    } catch (error) {
        // Handle CORS or network errors
        if (error.message.includes('CORS') || error.message.includes('NetworkError')) {
            throw new Error('Unable to fetch image. Try uploading from the extension popup instead.');
        }
        throw error;
    }
}

// Add visual indicator when hovering over images (optional enhancement)
let isEnabled = true;

chrome.storage.local.get(['extensionEnabled'], (data) => {
    isEnabled = data.extensionEnabled !== false; // Default to true
});

// Optional: Add hover effect to images to indicate they can be checked
if (isEnabled) {
    document.addEventListener('mouseover', (e) => {
        if (e.target.tagName === 'IMG' && !e.target.dataset.deepguardHover) {
            e.target.dataset.deepguardHover = 'true';
            e.target.style.cursor = 'pointer';
            e.target.title = 'Right-click to check for deepfake';
        }
    }, true);
}
