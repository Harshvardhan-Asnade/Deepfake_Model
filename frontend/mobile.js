/**
 * Mobile Navigation & Utilities
 * Handles hamburger menu, touch events, and mobile-specific optimizations
 */

(function () {
    'use strict';

    // ==================== HAMBURGER MENU ====================
    const hamburger = document.getElementById('hamburger');
    const navMenuWrapper = document.querySelector('.nav-menu-wrapper');
    const body = document.body;

    if (hamburger && navMenuWrapper) {
        // Toggle menu
        hamburger.addEventListener('click', function () {
            this.classList.toggle('active');
            navMenuWrapper.classList.toggle('active');
            body.classList.toggle('menu-open');
        });

        // Close menu when clicking on a nav link
        const navLinks = navMenuWrapper.querySelectorAll('.nav-menu a, .btn-primary');
        navLinks.forEach(link => {
            link.addEventListener('click', function () {
                hamburger.classList.remove('active');
                navMenuWrapper.classList.remove('active');
                body.classList.remove('menu-open');
            });
        });

        // Close menu when clicking outside
        document.addEventListener('click', function (event) {
            const isClickInsideNav = navMenuWrapper.contains(event.target);
            const isClickOnHamburger = hamburger.contains(event.target);

            if (!isClickInsideNav && !isClickOnHamburger && navMenuWrapper.classList.contains('active')) {
                hamburger.classList.remove('active');
                navMenuWrapper.classList.remove('active');
                body.classList.remove('menu-open');
            }
        });

        // Close menu on ESC key
        document.addEventListener('keydown', function (event) {
            if (event.key === 'Escape' && navMenuWrapper.classList.contains('active')) {
                hamburger.classList.remove('active');
                navMenuWrapper.classList.remove('active');
                body.classList.remove('menu-open');
            }
        });
    }

    // ==================== VIEWPORT HEIGHT FIX (iOS) ====================
    // Fix for 100vh on mobile browsers (address bar issue)
    function setViewportHeight() {
        const vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
    }

    setViewportHeight();
    window.addEventListener('resize', setViewportHeight);
    window.addEventListener('orientationchange', setViewportHeight);

    // ==================== TOUCH IMPROVEMENTS ====================
    // Add touch-active class for better touch feedback
    document.querySelectorAll('button, a, .tech-card, .showcase-item, .history-card').forEach(element => {
        element.addEventListener('touchstart', function () {
            this.classList.add('touch-active');
        }, { passive: true });

        element.addEventListener('touchend', function () {
            this.classList.remove('touch-active');
        }, { passive: true });

        element.addEventListener('touchcancel', function () {
            this.classList.remove('touch-active');
        }, { passive: true });
    });

    // ==================== PREVENT ZOOM ON INPUT FOCUS ====================
    // Already handled in CSS with font-size: 16px, but adding for completeness
    const inputs = document.querySelectorAll('input, textarea, select');
    inputs.forEach(input => {
        input.addEventListener('focus', function () {
            const viewport = document.querySelector('meta[name=viewport]');
            if (viewport) {
                viewport.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0';
            }
        });

        input.addEventListener('blur', function () {
            const viewport = document.querySelector('meta[name=viewport]');
            if (viewport) {
                viewport.content = 'width=device-width, initial-scale=1.0';
            }
        });
    });

    // ==================== MOBILE DETECTION ====================
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const isTablet = /(iPad|tablet|(android(?!.*mobile))|(windows(?!.*phone)(.*touch))|kindle|playbook|silk|(puffin(?!.*(IP|AP|WP))))/.test(navigator.userAgent.toLowerCase());

    if (isMobile) {
        document.body.classList.add('is-mobile');
    }
    if (isTablet) {
        document.body.classList.add('is-tablet');
    }

    // ==================== SMOOTH SCROLL POLYFILL ====================
    // For browsers that don't support smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // ==================== DEBOUNCED RESIZE HANDLER ====================
    let resizeTimer;
    window.addEventListener('resize', function () {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function () {
            // Trigger custom event that other scripts can listen to
            window.dispatchEvent(new CustomEvent('debouncedResize'));
        }, 250);
    });

    // ==================== LAZY LOAD OPTIMIZATION ====================
    // Only load images when they're about to enter the viewport
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                        observer.unobserve(img);
                    }
                }
            });
        }, {
            rootMargin: '50px'
        });

        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }

    // ==================== PREVENT OVERSCROLL (iOS) ====================
    // Prevent rubber-band scrolling on iOS
    let scrollStartY = 0;

    document.addEventListener('touchstart', function (e) {
        scrollStartY = e.touches[0].pageY;
    }, { passive: true });

    document.addEventListener('touchmove', function (e) {
        const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
        const scrollHeight = document.documentElement.scrollHeight;
        const clientHeight = document.documentElement.clientHeight;
        const scrollY = e.touches[0].pageY;

        // Prevent overscroll at top
        if (scrollTop === 0 && scrollY > scrollStartY) {
            e.preventDefault();
        }

        // Prevent overscroll at bottom
        if (scrollTop + clientHeight >= scrollHeight && scrollY < scrollStartY) {
            e.preventDefault();
        }
    }, { passive: false });

    // ==================== PERFORMANCE OPTIMIZATION ====================
    // Reduce animations on low-end devices
    if (navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4) {
        document.body.classList.add('reduce-motion');
    }

    // Detect slow connection
    if ('connection' in navigator) {
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
        if (connection && (connection.effectiveType === '2g' || connection.effectiveType === 'slow-2g')) {
            document.body.classList.add('slow-connection');
            // Disable heavy animations
            document.querySelectorAll('.floating-3d-object').forEach(el => {
                el.style.display = 'none';
            });
        }
    }

    // ==================== HORIZONTAL SCROLL INDICATOR ====================
    // Add scroll indicator for tables on mobile
    const scrollableElements = document.querySelectorAll('.history-table-container, .pipeline');
    scrollableElements.forEach(element => {
        if (element.scrollWidth > element.clientWidth) {
            element.classList.add('has-horizontal-scroll');

            // Remove indicator after first scroll
            element.addEventListener('scroll', function () {
                this.classList.remove('has-horizontal-scroll');
            }, { once: true });
        }
    });

    // ==================== STATUS BAR COLOR (PWA) ====================
    // Set theme color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name=theme-color]');
    if (!metaThemeColor) {
        const meta = document.createElement('meta');
        meta.name = 'theme-color';
        meta.content = '#000000';
        document.head.appendChild(meta);
    }

    console.log('🚀 Mobile optimizations loaded');
})();
