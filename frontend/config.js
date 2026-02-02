const CONFIG = {
    // API Base URL - Automatically selects between Localhost and Production
    API_BASE_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:7860'
        : 'https://harshasnade-deepfake-detection.hf.space'
};
