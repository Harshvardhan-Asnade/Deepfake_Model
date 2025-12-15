/**
 * DeepGuard Browser Extension - Background Service Worker
 * Handles context menu creation and message passing
 */

const API_URL = 'http://localhost:5001/api/predict';

// Create context menu on installation
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: 'checkDeepfake',
        title: 'Check for Deepfake',
        contexts: ['image']
    });

    console.log('DeepGuard extension installed successfully!');
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === 'checkDeepfake' && info.srcUrl) {

        // 1. Set analyzing state immediately
        await chrome.storage.local.set({
            analyzing: true,
            analyzingUrl: info.srcUrl,
            lastError: null,
            showLastResult: false
        });

        // 2. Open Popup immediately (Requires User Gesture, which this event has)
        try {
            await chrome.action.openPopup();
        } catch (e) {
            console.warn('Could not open popup automatically:', e);
            // If popup fails to open (e.g. Chrome restriction), we'll rely on notification
        }

        // 3. Perform Analysis in Background
        analyzeImage(info.srcUrl);
    }
});

// Analyze Image Function
async function analyzeImage(imageUrl) {
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
            // Success
            handleSuccess(data, imageUrl);
        } else {
            throw new Error(data.error || 'API request failed');
        }

    } catch (error) {
        console.error('Analysis failed:', error);
        handleError(error.message);
    }
}

// Handle Success
function handleSuccess(result, imageUrl) {
    // Add to history
    chrome.storage.local.get(['history'], (data) => {
        const history = data.history || [];
        history.unshift({
            timestamp: Date.now(),
            imageUrl: imageUrl,
            result: result,
            source: 'context-menu'
        });

        if (history.length > 20) history.pop();

        // Update Storage
        chrome.storage.local.set({
            history,
            analyzing: false,
            lastResult: result,
            showLastResult: true
        });

        // Notification fallback
        showNotification(result);
    });
}

// Handle Error
function handleError(errorMessage) {
    chrome.storage.local.set({
        analyzing: false,
        lastError: errorMessage
    });
}

// Show notification with result
function showNotification(result) {
    const isReal = result.prediction === 'REAL';
    const title = isReal ? '✓ Image appears Real' : '⚠ Deepfake Detected';
    const confidence = (result.confidence * 100).toFixed(1);
    const message = `Confidence: ${confidence}%`;

    console.log(`${title} - ${message}`);
}

// Handle messages from popup (if needed)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'clearHistory') {
        chrome.storage.local.set({ history: [] }, () => {
            sendResponse({ success: true });
        });
        return true;
    }
});
