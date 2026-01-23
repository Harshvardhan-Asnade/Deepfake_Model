// ==================== ENHANCED LOADER SYSTEM ====================
// Countdown Timer and Real/Fake Status Cycling (Minimum 5 seconds)

const initLoader = () => {
    console.log("Initializing Loader...");
    const loaderWrapper = document.getElementById('loader-wrapper');
    const loaderPercent = document.getElementById('loader-percent');
    const detectionStatus = document.getElementById('detectionStatus');
    const statusText = detectionStatus?.querySelector('.status-text');
    const loaderTimestamp = document.getElementById('loaderTimestamp');
    const loaderQuote = document.getElementById('loader-quote');

    if (!loaderWrapper) {
        console.error("Loader wrapper not found!");
        return;
    }

    // Force visibility at start
    loaderWrapper.style.display = 'flex';
    loaderWrapper.style.opacity = '1';

    // Prevent double initialization
    if (loaderWrapper.dataset.initialized) return;
    loaderWrapper.dataset.initialized = "true";

    let progress = 0;
    const targetProgress = 100;
    let currentStatus = 0;
    const startTime = Date.now();
    const minimumDuration = 3500; // 3.5 seconds (Optimized for better UX)

    // Safety Timeout - Force remove loader after 7 seconds if it gets stuck
    setTimeout(() => {
        if (loaderWrapper && loaderWrapper.style.display !== 'none' && !loaderWrapper.classList.contains('loaded')) {
            console.warn("Loader safety timeout triggered - forcing removal.");
            loaderWrapper.classList.add('loaded');
            setTimeout(() => {
                loaderWrapper.style.display = 'none';
            }, 800);
        }
    }, 7000);

    // Real/Fake Status Messages
    const statusMessages = [
        { text: 'ANALYZING...', class: 'analyzing' },
        { text: 'SCANNING PATTERNS...', class: 'analyzing' },
        { text: 'REAL?', class: 'real' },
        { text: 'CHECKING AUTHENTICITY...', class: 'analyzing' },
        { text: 'FAKE?', class: 'fake' },
        { text: 'VERIFYING DATA...', class: 'analyzing' }
    ];

    // Update timestamp
    const updateTimestamp = () => {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
        if (loaderTimestamp) {
            loaderTimestamp.textContent = `SYSTEM TIME: ${timeStr}`;
        }
    };
    updateTimestamp();
    const timeInterval = setInterval(updateTimestamp, 1000);

    // Cycle through status messages
    const cycleStatus = () => {
        if (!statusText || !detectionStatus) return;

        const status = statusMessages[currentStatus];
        statusText.textContent = status.text;

        // Remove all status classes
        detectionStatus.classList.remove('analyzing', 'real', 'fake');
        // Add current class
        detectionStatus.classList.add(status.class);

        currentStatus = (currentStatus + 1) % statusMessages.length;
    };

    // Start cycling status every 800ms
    cycleStatus();
    const statusInterval = setInterval(cycleStatus, 800);

    // Countdown Timer Animation - Smooth progression from 0 to 100
    let lastFrameTime = startTime;

    // Pre-process loader quotes for typing effect
    let allChars = [];
    if (loaderQuote && !loaderQuote.dataset.processed) {
        loaderQuote.dataset.processed = "true";

        const processNode = (node) => {
            if (node.nodeType === Node.TEXT_NODE) {
                const text = node.textContent;
                // Skip empty text nodes that are just whitespace to avoid weird spacing gaps if flex/grid were used, 
                // but for standard flow, whitespace is needed. However, large blocks of whitespace can be ignored.
                if (text.trim().length === 0 && text.length > 0) {
                    // Keep the whitespace node as is
                    return;
                }

                const fragment = document.createDocumentFragment();
                const map = text.split('');
                map.forEach(char => {
                    const span = document.createElement('span');
                    span.textContent = char;
                    span.className = 'char-waiting'; // Start in waiting state
                    fragment.appendChild(span);
                    allChars.push(span);
                });
                node.replaceWith(fragment);
            } else if (node.nodeType === Node.ELEMENT_NODE) {
                if (node.tagName !== 'BR') {
                    Array.from(node.childNodes).forEach(processNode);
                }
            }
        };

        Array.from(loaderQuote.childNodes).forEach(processNode);

        // Reveal the quote container after processing spans
        // Synchronous update to prevent frame flicker
        loaderQuote.style.opacity = '1';
    }

    function animateLoader() {
        const now = Date.now();
        const elapsed = now - startTime;

        // Calculate exact progress based on time (0 to 100 over 5 seconds)
        const exactProgress = Math.min((elapsed / minimumDuration) * 100, 100);

        // Update progress value directly (no interpolation to avoid jumps)
        progress = exactProgress;

        if (loaderPercent) {
            loaderPercent.textContent = Math.floor(progress);
        }

        if (allChars.length > 0) {
            const totalChars = allChars.length;
            // Calculate how many characters should be lit up based on progress
            // We want all chars lit by 100%
            const charsToLight = Math.floor((progress / 100) * totalChars);

            allChars.forEach((charSpan, index) => {
                if (index < charsToLight) {
                    charSpan.className = 'char-typed';
                } else if (index === charsToLight && index < totalChars) {
                    charSpan.className = 'char-current';
                } else {
                    charSpan.className = 'char-waiting';
                }
            });
        } else if (loaderQuote) {
            // Fallback if processing failed
            loaderQuote.style.opacity = progress / 100;
        }

        // Only finish when minimum time has elapsed and we're at 100%
        if (elapsed >= minimumDuration && progress >= 100) {
            // Ensure we show 100%
            if (loaderPercent) {
                loaderPercent.textContent = '100';
            }

            // Reached 100% - Exit loader
            setTimeout(() => {
                console.log("Loader finished.");
                clearInterval(statusInterval);
                clearInterval(timeInterval);

                loaderWrapper.classList.add('loaded'); // CSS transform

                // Remove from DOM after animation
                setTimeout(() => {
                    loaderWrapper.style.display = 'none';
                }, 800); // Wait for CSS transition
            }, 500); // Brief pause at 100%
        } else {
            requestAnimationFrame(animateLoader);
        }
    }

    // Start countdown
    animateLoader();
};

// Robust initialization
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initLoader);
} else {
    // DOM already ready, run immediately
    initLoader();
}
