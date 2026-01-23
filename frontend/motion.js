/**
 * DeepGuard Motion Design System
 * Implements: Lenis Smooth Scroll, Magnetic Buttons, Spotlight Cards, Text Reveals
 */

document.addEventListener('DOMContentLoaded', () => {
    const lenis = initSmoothScroll();
    initMagneticButtons();
    initSpotlightCards();
    initTextReveals();
    if (lenis) {
        initParallax(lenis);
    }
});

/* ==================== 1. SMOOTH SCROLL (LENIS) ==================== */
function initSmoothScroll() {
    // Check if Lenis is loaded
    if (typeof Lenis === 'undefined') {
        console.warn('Lenis not loaded. Skipping smooth scroll.');
        return null;
    }

    const lenis = new Lenis({
        duration: 1.2,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        direction: 'vertical',
        gestureDirection: 'vertical',
        smooth: true,
        mouseMultiplier: 1,
        smoothTouch: false,
        touchMultiplier: 2,
    });

    function raf(time) {
        lenis.raf(time);
        requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    return lenis;
}

/* ==================== 2. MAGNETIC BUTTONS ==================== */
function initMagneticButtons() {
    const buttons = document.querySelectorAll('.btn-primary, .btn-hero-primary');

    buttons.forEach(btn => {
        btn.addEventListener('mousemove', (e) => {
            const rect = btn.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Calculate distance from center
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const deltaX = (x - centerX) * 0.3; // Strength of pull
            const deltaY = (y - centerY) * 0.3;

            btn.style.transform = `translate(${deltaX}px, ${deltaY}px)`;
        });

        btn.addEventListener('mouseleave', () => {
            btn.style.transform = 'translate(0px, 0px)';
        });
    });
}

/* ==================== 3. SPOTLIGHT CARDS ==================== */
function initSpotlightCards() {
    const cards = document.querySelectorAll('.feature-card, .showcase-item, .tech-card');

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);
        });
    });
}

/* ==================== 4. TEXT REVEALS ==================== */
function initTextReveals() {
    // Targets: Hero title, Section titles
    const targets = document.querySelectorAll('.hero-title, .section-title');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
                observer.unobserve(entry.target); // Only animate once
            }
        });
    }, {
        threshold: 0.2
    });

    targets.forEach(target => {
        observer.observe(target);
    });
}

/* ==================== 5. PARALLAX EFFECTS ==================== */
function initParallax(lenis) {
    const parallaxItems = document.querySelectorAll('[data-speed]');

    if (parallaxItems.length === 0) return;

    lenis.on('scroll', ({ scroll }) => {
        parallaxItems.forEach(item => {
            const speed = parseFloat(item.dataset.speed) || 0;
            // Apply standard translation
            // Note: This overrides other transforms, so use on dedicated wrappers or elements without other transforms
            item.style.transform = `translateY(${scroll * speed}px)`;
        });
    });
}
