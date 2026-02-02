/**
 * DeepGuard Browser Extension - Content Script
 * Handles image capture and API communication from web pages
 */

const API_URL = 'http://localhost:7860/api/predict';

// --- Styles for the Overlay Button ---
const style = document.createElement('style');
style.textContent = `
  .deepguard-overlay-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    z-index: 9999;
    background-color: #E3F514; /* Brand Yellow */
    color: #000;
    border: none;
    border-radius: 20px;
    padding: 6px 12px;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    opacity: 0;
    transform: translateY(-5px);
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 6px;
    pointer-events: auto;
  }

  .deepguard-overlay-btn:hover {
    transform: translateY(0) scale(1.05);
    box-shadow: 0 4px 12px rgba(227, 245, 20, 0.4);
  }

  /* Show button when hovering the container or the button itself */
  .deepguard-image-wrapper:hover .deepguard-overlay-btn,
  .deepguard-overlay-btn:hover {
    opacity: 1;
    transform: translateY(0);
  }
  
  /* Instagram/Facebook specific adjustments */
  .deepguard-image-wrapper {
    position: relative;
    display: inline-block;
  }
`;
document.head.appendChild(style);

// --- Image Processing Logic ---

function addOverlayToImage(img) {
    if (img.dataset.deepguardProcessed) return;

    // Filter out small icons/nav elements
    const minSize = 150;
    const rect = img.getBoundingClientRect();
    if (rect.width < minSize || rect.height < minSize) return;

    // Use a wrapper to position the button relative to the image
    // Note: On some complex sites like IG, modifying DOM structure might break layout.
    // Instead of wrapping, we can try to position absolute relative to offsetParent.
    // or insert the button as a sibling if the parent is relatively positioned.

    // Strategy: Insert button as sibling, ensure parent is relative.
    const parent = img.parentElement;
    if (!parent) return;

    const computedStyle = window.getComputedStyle(parent);
    if (computedStyle.position === 'static') {
        parent.style.position = 'relative';
    }

    const btn = document.createElement('button');
    btn.className = 'deepguard-overlay-btn';
    btn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
           <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
           <polyline points="22 4 12 14.01 9 11.01"></polyline>
        </svg>
        Check
    `;

    btn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const imageUrl = img.src || img.srcset?.split(' ')[0]; // Handle srcset
        if (imageUrl) {
            sendImageForAnalysis(imageUrl);
        }
    });

    // Make sure we don't mess up the layout
    img.dataset.deepguardProcessed = 'true';
    parent.classList.add('deepguard-image-wrapper');
    parent.appendChild(btn);
}

function processImages() {
    const images = document.querySelectorAll('img');
    images.forEach(addOverlayToImage);

    // Handle Instagram/Facebook div-background images if needed
    // (Advanced: would require analyzing computed background-image)
}

// --- Mutation Observer for Dynamic Content ---

const observer = new MutationObserver((mutations) => {
    let shouldProcess = false;
    for (const mutation of mutations) {
        if (mutation.addedNodes.length > 0) {
            shouldProcess = true;
            break;
        }
    }
    if (shouldProcess) {
        processImages();
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// Run initially
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', processImages);
} else {
    processImages();
}


// --- API Communication ---

async function sendImageForAnalysis(imageUrl) {
    // Notify background script to handle the analysis (avoids CORS issues in content script sometimes, 
    // but we can also try direct fetch if CORs allows. Background is safer.)

    // We'll use the background script to open the popup or show notification
    // But actually, we want to see the result IN the page or popup.
    // Let's send a message to background to set "analyzing" state and open popup.

    chrome.runtime.sendMessage({
        action: 'analyzeImage',
        imageUrl: imageUrl,
        openPopup: true
    }, (response) => {
        if (chrome.runtime.lastError) {
            console.error("Runtime error:", chrome.runtime.lastError);
            alert("DeepGuard Error: Connection to background script failed.");
        }
    });
}
