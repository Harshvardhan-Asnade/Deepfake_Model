/**
 * DeepGuard Service Worker
 * Handles offline functionality, caching, and PWA features
 */

const CACHE_VERSION = 'v1.0.0';
const CACHE_NAME = `deepguard-${CACHE_VERSION}`;

// Assets to cache immediately on install
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/analysis.html',
    '/history.html',
    '/offline.html',
    '/style.css',
    '/animations.css',
    '/history.css',
    '/loader.css',
    '/orbit.css',
    '/showcase.css',
    '/video_player.css',
    '/extension.css',
    '/scroll_indicator.css',
    '/script.js',
    '/mobile.js',
    '/loader.js',
    '/hero_reveal.js',
    '/motion.js',
    '/orbit_interaction.js',
    '/three_bg.js',
    '/logo.ico',
    '/icon-192.png',
    '/icon-512.png',
    '/manifest.json'
];

// Assets that can be cached on demand
const RUNTIME_CACHE = 'deepguard-runtime';

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('[Service Worker] Installing...');

    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[Service Worker] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => {
                console.log('[Service Worker] Installation complete');
                return self.skipWaiting(); // Activate immediately
            })
            .catch((error) => {
                console.error('[Service Worker] Installation failed:', error);
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('[Service Worker] Activating...');

    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => name.startsWith('deepguard-') && name !== CACHE_NAME)
                        .map((name) => {
                            console.log('[Service Worker] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => {
                console.log('[Service Worker] Activation complete');
                return self.clients.claim(); // Take control immediately
            })
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip cross-origin requests
    if (url.origin !== location.origin) {
        return;
    }

    // Skip API requests (let them go to network)
    if (url.pathname.startsWith('/api/')) {
        return;
    }

    event.respondWith(
        caches.match(request)
            .then((cachedResponse) => {
                if (cachedResponse) {
                    console.log('[Service Worker] Serving from cache:', request.url);
                    return cachedResponse;
                }

                // Not in cache, fetch from network
                return fetch(request)
                    .then((response) => {
                        // Don't cache non-successful responses
                        if (!response || response.status !== 200 || response.type !== 'basic') {
                            return response;
                        }

                        // Clone response for caching
                        const responseToCache = response.clone();

                        // Cache runtime assets
                        caches.open(RUNTIME_CACHE)
                            .then((cache) => {
                                cache.put(request, responseToCache);
                            });

                        return response;
                    })
                    .catch((error) => {
                        console.error('[Service Worker] Fetch failed:', error);

                        // Return offline page for navigation requests
                        if (request.mode === 'navigate') {
                            return caches.match('/offline.html');
                        }

                        // Return fallback for images
                        if (request.destination === 'image') {
                            return caches.match('/logo.svg');
                        }

                        return new Response('Offline - content not available', {
                            status: 503,
                            statusText: 'Service Unavailable',
                            headers: new Headers({
                                'Content-Type': 'text/plain'
                            })
                        });
                    });
            })
    );
});

// Background sync for offline analysis (future enhancement)
self.addEventListener('sync', (event) => {
    console.log('[Service Worker] Background sync:', event.tag);

    if (event.tag === 'sync-analyses') {
        event.waitUntil(syncPendingAnalyses());
    }
});

// Push notifications (future enhancement)
self.addEventListener('push', (event) => {
    console.log('[Service Worker] Push notification received');

    const data = event.data ? event.data.json() : {};
    const title = data.title || 'DeepGuard';
    const options = {
        body: data.body || 'Analysis complete',
        icon: '/icon-192.png',
        badge: '/icon-192.png',
        vibrate: [200, 100, 200],
        data: {
            url: data.url || '/history.html'
        }
    };

    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
    console.log('[Service Worker] Notification clicked');
    event.notification.close();

    const urlToOpen = event.notification.data.url || '/';

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then((windowClients) => {
                // Check if there's already a window open
                for (let client of windowClients) {
                    if (client.url === urlToOpen && 'focus' in client) {
                        return client.focus();
                    }
                }
                // Open new window
                if (clients.openWindow) {
                    return clients.openWindow(urlToOpen);
                }
            })
    );
});

// Helper function for background sync
async function syncPendingAnalyses() {
    // Future implementation: sync pending analyses to backend
    console.log('[Service Worker] Syncing pending analyses...');
    return Promise.resolve();
}

// Message handler for skip waiting
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});

console.log('[Service Worker] Loaded successfully');
