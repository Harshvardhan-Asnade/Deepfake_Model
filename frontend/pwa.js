/**
 * PWA Installation and Service Worker Registration
 */

(function () {
    'use strict';

    // ==================== SERVICE WORKER REGISTRATION ====================
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', () => {
            navigator.serviceWorker.register('/service-worker.js')
                .then((registration) => {
                    console.log('✅ Service Worker registered:', registration.scope);

                    // Check for updates
                    registration.addEventListener('updatefound', () => {
                        const newWorker = registration.installing;
                        console.log('🔄 Service Worker update found');

                        newWorker.addEventListener('statechange', () => {
                            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                                // New version available
                                showUpdateNotification();
                            }
                        });
                    });
                })
                .catch((error) => {
                    console.error('❌ Service Worker registration failed:', error);
                });
        });
    }

    // ==================== PWA INSTALL PROMPT ====================
    let deferredPrompt;
    let installButton;

    // Listen for install prompt event
    window.addEventListener('beforeinstallprompt', (e) => {
        console.log('💾 Install prompt triggered');

        // Prevent Chrome 67 and earlier from automatically showing the prompt
        e.preventDefault();

        // Store the event for later use
        deferredPrompt = e;

        // Show install button
        showInstallButton();
    });

    // Create and show install button
    function showInstallButton() {
        // Check if already installed
        if (window.matchMedia('(display-mode: standalone)').matches) {
            console.log('Already installed as PWA');
            return;
        }

        // Check if button already exists
        if (document.getElementById('pwa-install-btn')) return;

        // Create install button
        installButton = document.createElement('button');
        installButton.id = 'pwa-install-btn';
        installButton.className = 'pwa-install-button';
        installButton.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            <span>Install App</span>
        `;

        installButton.addEventListener('click', handleInstallClick);

        // Add to page
        document.body.appendChild(installButton);

        // Fade in animation
        setTimeout(() => {
            installButton.classList.add('visible');
        }, 100);
    }

    // Handle install button click
    async function handleInstallClick() {
        if (!deferredPrompt) return;

        // Show install prompt
        deferredPrompt.prompt();

        // Wait for user choice
        const { outcome } = await deferredPrompt.userChoice;

        console.log(`User response to install prompt: ${outcome}`);

        if (outcome === 'accepted') {
            console.log('✅ PWA installed');
            hideInstallButton();
        } else {
            console.log('❌ PWA installation declined');
        }

        // Clear the deferredPrompt
        deferredPrompt = null;
    }

    // Hide install button
    function hideInstallButton() {
        if (installButton) {
            installButton.classList.remove('visible');
            setTimeout(() => {
                if (installButton && installButton.parentNode) {
                    installButton.remove();
                }
            }, 300);
        }
    }

    // ==================== DETECT PWA MODE ====================
    window.addEventListener('DOMContentLoaded', () => {
        // Check if running as installed PWA
        const isStandalone = window.matchMedia('(display-mode: standalone)').matches ||
            window.navigator.standalone ||
            document.referrer.includes('android-app://');

        if (isStandalone) {
            console.log('🚀 Running as PWA');
            document.body.classList.add('pwa-mode');

            // Add iOS status bar spacing
            if (navigator.userAgent.match(/iPhone|iPad|iPod/)) {
                document.body.classList.add('ios-pwa');
            }
        } else {
            console.log('🌐 Running in browser');
        }
    });

    // ==================== UPDATE NOTIFICATION ====================
    function showUpdateNotification() {
        // Check if notification already exists
        if (document.getElementById('pwa-update-notification')) return;

        const notification = document.createElement('div');
        notification.id = 'pwa-update-notification';
        notification.className = 'pwa-update-notification';
        notification.innerHTML = `
            <div class="update-content">
                <div class="update-icon">🔄</div>
                <div class="update-text">
                    <strong>New version available!</strong>
                    <p>Click to update and get the latest features</p>
                </div>
                <button class="update-btn" onclick="window.location.reload()">
                    Update Now
                </button>
                <button class="update-dismiss" onclick="this.parentElement.parentElement.remove()">
                    ✕
                </button>
            </div>
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.classList.add('visible');
        }, 100);
    }

    // ==================== ONLINE/OFFLINE STATUS ====================
    window.addEventListener('online', () => {
        console.log('🌐 Connection restored');
        showToast('Back online!', 'success');
    });

    window.addEventListener('offline', () => {
        console.log('📡 Connection lost');
        showToast('You are offline. Some features may be limited.', 'warning');
    });

    // Helper function for toast (if not already defined)
    function showToast(message, type = 'info') {
        // Use existing toast function if available
        if (window.showToast) {
            window.showToast(message, type);
            return;
        }

        // Simple fallback toast
        const toast = document.createElement('div');
        toast.className = `simple-toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${type === 'success' ? '#10b981' : type === 'warning' ? '#f59e0b' : '#3b82f6'};
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            z-index: 10000;
            animation: slideUp 0.3s ease;
        `;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

    // ==================== iOS ADD TO HOME SCREEN PROMPT ====================
    function showIOSInstallPrompt() {
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
        const isInStandaloneMode = ('standalone' in window.navigator) && (window.navigator.standalone);

        if (isIOS && !isInStandaloneMode) {
            // Check if user has seen this before
            if (localStorage.getItem('ios-install-prompt-dismissed')) {
                return;
            }

            const prompt = document.createElement('div');
            prompt.className = 'ios-install-prompt';
            prompt.innerHTML = `
                <div class="ios-prompt-content">
                    <button class="ios-prompt-close" onclick="this.parentElement.parentElement.remove(); localStorage.setItem('ios-install-prompt-dismissed', 'true');">✕</button>
                    <div class="ios-prompt-icon">
                        <img src="/logo.ico" alt="DeepGuard" style="width: 60px; height: 60px; border-radius: 12px;">
                    </div>
                    <h3>Install DeepGuard</h3>
                    <p>Install this app on your iPhone:</p>
                    <ol>
                        <li>Tap the <strong>Share</strong> button <svg width="16" height="20" viewBox="0 0 16 20" fill="#0066cc"><path d="M8 0l8 8h-5v12H5V8H0z"/></svg></li>
                        <li>Select <strong>Add to Home Screen</strong></li>
                    </ol>
                </div>
            `;
            document.body.appendChild(prompt);

            setTimeout(() => {
                prompt.classList.add('visible');
            }, 2000);
        }
    }

    // Show iOS prompt after short delay
    setTimeout(showIOSInstallPrompt, 3000);

    console.log('📱 PWA features initialized');
})();
