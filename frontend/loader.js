// ==================== ENHANCED LOADER SYSTEM ====================
// Countdown Timer and Real/Fake Status Cycling (Minimum 5 seconds)

const initLoader = () => {
    console.log("Initializing Loader...");
    const loaderWrapper = document.getElementById('loader-wrapper');
    const loaderPercent = document.getElementById('loader-percent');
    const detectionStatus = document.getElementById('detectionStatus');
    const statusText = detectionStatus?.querySelector('.status-text');
    const loaderTimestamp = document.getElementById('loaderTimestamp');

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
    const minimumDuration = 2500; // 2.5 seconds (Optimized for better UX)

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
