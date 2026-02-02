/**
 * DeepGuard Browser Extension - Compact Popup Logic
 */

const API_URL = 'http://localhost:7860/api/predict';
const API_STATUS_URL = 'http://localhost:7860/api/health';

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const uploadSection = document.getElementById('uploadSection');
const fileInput = document.getElementById('fileInput');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');
const resultBadge = document.getElementById('resultBadge');
const resultLabel = document.getElementById('resultLabel');
const resultConfidence = document.getElementById('resultConfidence');
const statusValue = document.getElementById('statusValue');
const confidenceDetail = document.getElementById('confidenceDetail');
const newCheckBtn = document.getElementById('newCheckBtn');
const historyGrid = document.getElementById('historyGrid');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const apiStatus = document.getElementById('apiStatus');

// State
let selectedFile = null;
let selectedImageUrl = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    loadHistory();
    setupEventListeners();
    checkForContextMenuResult();
});

// Check if there's a result from context menu or active analysis
function checkForContextMenuResult() {
    chrome.storage.local.get(['showLastResult', 'lastResult', 'lastError', 'analyzing', 'analyzingUrl'], (data) => {

        // 1. Check if currently analyzing
        if (data.analyzing) {
            uploadSection.style.display = 'none';
            loadingState.style.display = 'block';
            resultsSection.style.display = 'none';
            if (data.analyzingUrl) {
                // Optional: Show preview of what is being analyzed
                // But we don't have the blob here, only URL.
            }
            return;
        }

        // 2. Check for completed result
        if (data.showLastResult && data.lastResult) {
            displayResults(data.lastResult, data.analyzingUrl || null); // Note: analyzingUrl might be stale, but history has it
            chrome.storage.local.set({ showLastResult: false }); // Consume it
        }

        // 3. Check for error
        else if (data.lastError) {
            showError(data.lastError);
            chrome.storage.local.remove('lastError');
        }
    });
}

// Listen for storage changes (background script updates)
chrome.storage.onChanged.addListener((changes, namespace) => {
    if (namespace === 'local') {
        if (changes.analyzing) {
            if (changes.analyzing.newValue) {
                // Started analyzing
                uploadSection.style.display = 'none';
                loadingState.style.display = 'block';
                resultsSection.style.display = 'none';
            }
        }

        if (changes.lastResult && changes.lastResult.newValue) {
            // Analysis finished successfully
            // Get the URL from storage since it should still be there or in history
            chrome.storage.local.get(['analyzingUrl'], (data) => {
                displayResults(changes.lastResult.newValue, data.analyzingUrl);
            });
        }

        if (changes.lastError && changes.lastError.newValue) {
            showError(changes.lastError.newValue);
        }
    }
});

// Setup Event Listeners
function setupEventListeners() {
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // Global Drag & Drop
    document.body.addEventListener('dragover', handleGlobalDragOver);
    document.body.addEventListener('dragleave', handleGlobalDragLeave);
    document.body.addEventListener('drop', handleGlobalDrop);

    newCheckBtn.addEventListener('click', resetToUpload);
    clearHistoryBtn.addEventListener('click', clearHistory);
}

// Check API Status
async function checkAPIStatus() {
    try {
        const response = await fetch(API_STATUS_URL);

        if (!response.ok) {
            throw new Error(`Health check failed: ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'healthy' && data.model_loaded) {
            apiStatus.classList.add('online');
            apiStatus.title = "System Online";
        } else {
            apiStatus.classList.add('offline');
            apiStatus.title = data.model_loaded ? "System Initializing" : "Model Not Loaded";
        }
    } catch (error) {
        console.error('API status check failed:', error);
        apiStatus.classList.add('offline');
        apiStatus.title = "Backend Offline - Start server on localhost:7860";
    }
}

// File Selection Handlers
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        selectedImageUrl = URL.createObjectURL(file);
        analyzeImage();
    }
}

// Global Drag & Drop Handlers
function handleGlobalDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.body.classList.add('dragging');
}

function handleGlobalDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    // Only remove if leaving the window (not just entering a child element)
    if (e.clientX <= 0 || e.clientY <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight) {
        document.body.classList.remove('dragging');
    }
}

function handleGlobalDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.body.classList.remove('dragging');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        selectedImageUrl = URL.createObjectURL(file);
        analyzeImage();
    }
}

// Analyze Image
async function analyzeImage() {
    if (!selectedFile) return;

    // Show loading state
    uploadSection.style.display = 'none';
    loadingState.style.display = 'block';
    resultsSection.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile); // Changed from 'image' to 'file' to match Flask

        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!data.error) {
            displayResults(data, selectedImageUrl);
            saveToHistory(data, selectedImageUrl);
        } else {
            showError(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        const errorMessage = error.message.includes('Failed to fetch')
            ? 'Unable to connect to the DeepGuard backend. Please ensure the backend is running on localhost:7860'
            : error.message;
        showError(errorMessage);
    }
}

// Display Results
function displayResults(prediction, imageUrl) {
    loadingState.style.display = 'none';
    uploadSection.style.display = 'none';
    resultsSection.style.display = 'flex';

    // Parse API response
    // API returns: { prediction: "REAL"|"FAKE", confidence: 0.xx, ... }
    const isReal = prediction.prediction === 'REAL';
    const confidencePercent = (prediction.confidence * 100).toFixed(1);

    // Set image
    if (imageUrl) {
        resultImage.src = imageUrl;
    }

    const imageContainer = document.querySelector('.result-image-container');

    // Visual Feedback
    if (isReal) {
        launchConfetti();
        imageContainer.classList.remove('fake-warning');
        // Green glow for real
        imageContainer.style.boxShadow = '0 4px 20px rgba(227, 245, 20, 0.3)';
        imageContainer.style.border = '2px solid var(--success)';
    } else {
        imageContainer.classList.add('fake-warning');
        imageContainer.style.boxShadow = 'none'; // Class handles it
        imageContainer.style.border = 'none'; // Class handles it
    }

    // Update badge
    resultLabel.textContent = isReal ? 'REAL' : 'FAKE';
    resultLabel.className = 'result-label ' + (isReal ? 'real' : 'fake');
    resultConfidence.textContent = confidencePercent + '%';

    // Update details
    statusValue.textContent = isReal ? 'Real Image ✓' : 'Deepfake Detected ⚠';
    statusValue.style.color = isReal ? 'var(--success)' : 'var(--warning)';
    confidenceDetail.textContent = confidencePercent + '%';
}

// Show Error
function showError(message) {
    loadingState.style.display = 'none';
    uploadSection.style.display = 'flex';
    resultsSection.style.display = 'none';
    alert(message);
}

// Reset to Upload
function resetToUpload() {
    selectedFile = null;
    selectedImageUrl = null;
    fileInput.value = '';
    uploadSection.style.display = 'flex';
    loadingState.style.display = 'none';
    resultsSection.style.display = 'none';
}

// History Management
function saveToHistory(result, imageUrl) {
    chrome.storage.local.get(['history'], (data) => {
        const history = data.history || [];

        history.unshift({
            timestamp: Date.now(),
            imageUrl: imageUrl,
            result: result,
            source: 'popup'
        });

        if (history.length > 12) {
            history.pop();
        }

        chrome.storage.local.set({ history }, () => {
            loadHistory();
        });
    });
}

function loadHistory() {
    chrome.storage.local.get(['history'], (data) => {
        const history = data.history || [];

        if (history.length === 0) {
            historyGrid.innerHTML = '<div class="empty-history">No recent checks</div>';
            return;
        }

        historyGrid.innerHTML = history.slice(0, 8).map(item => {
            const isReal = item.result.prediction === 'REAL';
            return `
        <div class="history-item" data-timestamp="${item.timestamp}" title="${new Date(item.timestamp).toLocaleString()}">
          <img src="${item.imageUrl}" alt="History">
          <div class="history-item-badge ${isReal ? 'real' : 'fake'}"></div>
        </div>
      `;
        }).join('');

        // Add click handlers
        document.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', () => {
                const timestamp = parseInt(item.dataset.timestamp);
                const historyItem = history.find(h => h.timestamp === timestamp);
                if (historyItem) {
                    displayResults(historyItem.result, historyItem.imageUrl);
                }
            });
        });
    });
}

function clearHistory() {
    if (confirm('Clear all history?')) {
        chrome.storage.local.set({ history: [] }, () => {
            loadHistory();
        });
    }
}

// Simple Confetti
function launchConfetti() {
    const colors = ['#E3F514', '#FFFFFF', '#D1E300', '#FAFAFA', '#000000'];

    for (let i = 0; i < 40; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';

        // Random properties
        const left = Math.random() * 100;
        const color = colors[Math.floor(Math.random() * colors.length)];
        const duration = Math.random() * 1.5 + 1.5; // 1.5-3s
        const delay = Math.random() * 0.5;

        confetti.style.left = left + '%';
        confetti.style.backgroundColor = color;
        confetti.style.animation = `fall ${duration}s ease-out ${delay}s forwards`;

        document.body.appendChild(confetti);

        // Cleanup
        setTimeout(() => {
            confetti.remove();
        }, (duration + delay) * 1000);
    }
}
