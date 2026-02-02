/**
 * DeepGuard Browser Extension - Background Service Worker
 * Handles context menu creation and message passing
 */

const API_URL = 'http://localhost:7860/api/predict';

// Create context menu on installation
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: 'checkDeepfake',
        title: 'Check for Deepfake',
        contexts: ['image']
    });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === 'checkDeepfake' && info.srcUrl) {
        startAnalysis(info.srcUrl);
    }
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'analyzeImage') {
        startAnalysis(request.imageUrl);
        sendResponse({ success: true, status: 'analysis_started' });
    } else if (request.action === 'clearHistory') {
        chrome.storage.local.set({ history: [] }, () => {
            sendResponse({ success: true });
        });
        return true;
    }
});

async function startAnalysis(imageUrl) {
    // 1. Set analyzing state immediately
    await chrome.storage.local.set({
        analyzing: true,
        analyzingUrl: imageUrl,
        lastError: null,
        showLastResult: false
    });

    // 2. Open Popup immediately (User experience improvement)
    try {
        await chrome.action.openPopup();
    } catch (e) {
        // Popups can only be opened by user gesture (click).
        // If triggered by content script button click, it MIGHT work if the browser passes the gesture.
        // If not, we rely on the badge or notification.
        console.warn('Could not open popup automatically (likely connection restriction):', e);

        // Set a badge to indicate activity
        chrome.action.setBadgeText({ text: '...' });
        chrome.action.setBadgeBackgroundColor({ color: '#E3F514' });
    }

    // 3. Perform Analysis
    analyzeImage(imageUrl);
}

// Analyze Image Function
async function analyzeImage(imageUrl) {
    try {
        console.log('Fetching:', imageUrl);

        // Fetch the image with proper CORS handling
        const response = await fetch(imageUrl, {
            mode: 'cors',
            credentials: 'omit'
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
        }

        const blob = await response.blob();

        // Validate that we got an image
        if (!blob.type.startsWith('image/')) {
            throw new Error('The fetched content is not an image');
        }

        // Create FormData
        const formData = new FormData();
        formData.append('file', blob, 'image.png');

        // Send to API
        console.log('Sending to API:', API_URL);
        const apiResponse = await fetch(API_URL, {
            method: 'POST',
            body: formData,
            mode: 'cors'
        });

        if (!apiResponse.ok) {
            // Try to get error text
            const errorText = await apiResponse.text().catch(() => 'Unknown error');
            throw new Error(`API request failed: ${apiResponse.status} ${apiResponse.statusText} - ${errorText}`);
        }

        const data = await apiResponse.json();

        if (!data.error) {
            // Success
            handleSuccess(data, imageUrl);
        } else {
            throw new Error(data.error || 'API request failed');
        }

    } catch (error) {
        console.error('Analysis failed:', error);

        // Provide more specific error messages
        let errorMessage = error.message;
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            errorMessage = 'Backend disconnected. Is localhost:7860 running?';
        } else if (error.message.includes('CORS')) {
            errorMessage = 'CORS Blocked. Try right-click > Save as... then upload.';
        }

        handleError(errorMessage);
    } finally {
        // Clear badge
        chrome.action.setBadgeText({ text: '' });
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
            source: 'web'
        });

        if (history.length > 20) history.pop();

        // Update Storage
        chrome.storage.local.set({
            history,
            analyzing: false,
            lastResult: result,
            showLastResult: true
        });

        // Notification fallback if popup is closed
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
    const title = isReal ? '✓ Real Image' : '⚠ Deepfake Detected';
    const confidence = (result.confidence * 100).toFixed(1);

    // We can use Chrome Notifications API properly here if permission is granted
    // For now, console log is a placeholder, but let's try a system notification
    /*
    chrome.notifications.create({
        type: 'basic',
        iconUrl: isReal ? 'icons/icon48.png' : 'icons/icon48.png', 
        title: title,
        message: `Confidence: ${confidence}%`,
        priority: 2
    });
    */
    console.log(`${title} - ${confidence}%`);
}
