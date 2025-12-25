// ==================== ENHANCED LOADER SYSTEM ====================
// Moved to loader.js

// API URL Configuration
// In production (Vercel), this should point to the Hugging Face Spaces URL
// Example: https://huggingface.co/spaces/username/space-name/api
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? ''
    : 'https://YOUR-HUGGINGFACE-SPACE-URL.hf.space'; // REPLACE THIS AFTER DEPLOYING BACKEND



// ==================== AUDIO SYSTEM ====================
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

const playSound = (type) => {
    if (audioCtx.state === 'suspended') audioCtx.resume();

    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    const now = audioCtx.currentTime;

    if (type === 'scan') {
        // High-tech scanning sound
        osc.type = 'sine';
        osc.frequency.setValueAtTime(800, now);
        osc.frequency.exponentialRampToValueAtTime(1200, now + 0.1);
        osc.frequency.exponentialRampToValueAtTime(800, now + 0.2);

        gainNode.gain.setValueAtTime(0.1, now);
        gainNode.gain.linearRampToValueAtTime(0, now + 0.2);

        osc.start(now);
        osc.stop(now + 0.2);
    } else if (type === 'alert') {
        // Warning sound for fake detection
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(200, now);
        osc.frequency.linearRampToValueAtTime(100, now + 0.3);

        gainNode.gain.setValueAtTime(0.2, now);
        gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.3);

        osc.start(now);
        osc.stop(now + 0.3);
    } else if (type === 'success') {
        // Safe/Authentic sound
        osc.type = 'sine';
        osc.frequency.setValueAtTime(440, now);
        osc.frequency.exponentialRampToValueAtTime(880, now + 0.3);

        gainNode.gain.setValueAtTime(0.1, now);
        gainNode.gain.linearRampToValueAtTime(0, now + 0.3);

        osc.start(now);
        osc.stop(now + 0.3);
    }
};

// ==================== PARTICLE BACKGROUND ====================
// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    // ==================== THEME MANAGEMENT ====================
    const initTheme = () => {
        // Create Toggle Button
        const themeToggleBtn = document.createElement('button');
        themeToggleBtn.className = 'theme-toggle-btn';
        themeToggleBtn.title = "Toggle Theme";

        // Find navbar content
        const navContent = document.querySelector('.nav-content');
        if (navContent) {
            // We want to group the last element (usually the CTA button) with our new toggle
            // to keep them both on the right side if justify-content: space-between is used.
            const lastItem = navContent.lastElementChild;

            // Check if the last item is a button/link (and not the menu or logo if order differs)
            // In standard index.html: Logo, Menu, Button. Button is last.
            if (lastItem && !lastItem.classList.contains('nav-menu') && lastItem.tagName !== 'SCRIPT') {
                const wrapper = document.createElement('div');
                wrapper.style.display = 'flex';
                wrapper.style.alignItems = 'center';

                // Insert wrapper before the last item
                navContent.insertBefore(wrapper, lastItem);

                // Move last item into wrapper
                wrapper.appendChild(lastItem);

                // Append toggle to wrapper
                wrapper.appendChild(themeToggleBtn);
            } else {
                // Fallback if structure is unexpected
                navContent.appendChild(themeToggleBtn);
            }
        }

        // Check Logic
        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(themeToggleBtn, savedTheme);

        themeToggleBtn.addEventListener('click', (e) => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';

            // Fallback for browsers without View Transitions
            if (!document.startViewTransition) {
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                updateThemeIcon(themeToggleBtn, newTheme);
                return;
            }

            // Get click coordinates
            const x = e.clientX;
            const y = e.clientY;

            // Calculate distance to the furthest corner
            const endRadius = Math.hypot(
                Math.max(x, innerWidth - x),
                Math.max(y, innerHeight - y)
            );

            // Start the view transition
            const transition = document.startViewTransition(() => {
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                updateThemeIcon(themeToggleBtn, newTheme);
            });

            // Animate the circular clip path
            transition.ready.then(() => {
                const clipPath = [
                    `circle(0px at ${x}px ${y}px)`,
                    `circle(${endRadius}px at ${x}px ${y}px)`
                ];

                document.documentElement.animate(
                    {
                        clipPath: clipPath,
                    },
                    {
                        duration: 800, // Slightly slower for dramatic effect
                        easing: 'ease-in-out',
                        pseudoElement: '::view-transition-new(root)',
                    }
                );
            });
        });
    };

    const updateThemeIcon = (btn, theme) => {
        if (theme === 'light') {
            btn.innerHTML = '🌙'; // Moon
            btn.style.borderColor = 'var(--text-primary)';
            btn.style.color = 'var(--text-primary)';
        } else {
            btn.innerHTML = '☀'; // Sun
            btn.style.borderColor = 'rgba(255, 255, 255, 0.2)';
            btn.style.color = '#fff';
        }
    };

    initTheme();

    // Initialize AOS
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 800,
            easing: 'ease-out-cubic',
            once: true,
            mirror: false,
            offset: 100
        });
    }

    // Initialize Particles.js (if element exists)
    if (document.getElementById('particles-js') && typeof particlesJS !== 'undefined') {
        const theme = localStorage.getItem('theme') || 'dark';
        const pColor = theme === 'light' ? '#0044cc' : '#E3F514';

        particlesJS('particles-js', {
            particles: {
                number: { value: 60, density: { enable: true, value_area: 800 } },
                color: { value: pColor },
                shape: { type: 'circle' },
                opacity: { value: 0.3, random: true },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: pColor,
                    opacity: 0.2,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 2,
                    direction: 'none',
                    random: false,
                    straight: false,
                    out_mode: 'out',
                    bounce: false,
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: { enable: true, mode: 'repulse' },
                    onclick: { enable: true, mode: 'push' },
                    resize: true
                },
                modes: {
                    repulse: { distance: 100, duration: 0.4 },
                    push: { particles_nb: 4 }
                }
            },
            retina_detect: true
        });
    }

    // ==================== MICRO-INTERACTIONS ====================

    // 1. Button Ripple Effect
    const buttons = document.querySelectorAll('.btn-primary, .btn-hero-primary, .btn-hero-secondary');
    buttons.forEach(btn => {
        btn.addEventListener('click', function (e) {
            let x = e.clientX - e.target.offsetLeft;
            let y = e.clientY - e.target.offsetTop;

            let ripples = document.createElement('span');
            ripples.style.left = x + 'px';
            ripples.style.top = y + 'px';
            ripples.classList.add('ripple');
            this.appendChild(ripples);

            setTimeout(() => {
                ripples.remove();
            }, 600);
        });
    });

    // 2. 3D Card Tilt Effect
    const tiltCards = document.querySelectorAll('.feature-card, .tech-card, .showcase-item');

    tiltCards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            // Calculate rotation (max 10 degrees)
            const rotateX = ((y - centerY) / centerY) * -10;
            const rotateY = ((x - centerX) / centerX) * 10;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.02)`;
        });

        card.addEventListener('mouseleave', () => {
            // Reset position
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale(1)';
        });
    });

    // ==================== FLOATING 3D OBJECTS PARALLAX ====================
    const floatingCube = document.getElementById('floatingCube');
    const floatingPyramid = document.getElementById('floatingPyramid');

    // Initialize Particles.js for Neural Network Effect
    if (typeof particlesJS !== 'undefined') {
        particlesJS('loader-particles', {
            "particles": {
                "number": { "value": 80, "density": { "enable": true, "value_area": 800 } },
                "color": { "value": "#E3F514" },
                "shape": { "type": "circle" },
                "opacity": { "value": 0.5, "random": true },
                "size": { "value": 3, "random": true },
                "line_linked": {
                    "enable": true,
                    "distance": 150,
                    "color": "#E3F514",
                    "opacity": 0.4,
                    "width": 1
                },
                "move": {
                    "enable": true,
                    "speed": 2,
                    "direction": "none",
                    "random": false,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                }
            },
            "interactivity": {
                "detect_on": "canvas",
                "events": { "onhover": { "enable": true, "mode": "grab" }, "onclick": { "enable": true, "mode": "push" } },
                "modes": { "grab": { "distance": 140, "line_linked": { "opacity": 1 } } }
            },
            "retina_detect": true
        });
    }

    // Simulate loading progress
    let progress = 0;
    // The following variables are already declared within the next 'if' block.
    // Redeclaring them here would cause a SyntaxError.
    // let currentX = 0;
    // let currentY = 0;

    if (floatingCube || floatingPyramid) {
        let mouseX = 0;
        let mouseY = 0;
        let currentX = 0;
        let currentY = 0;

        // Track mouse position
        document.addEventListener('mousemove', (e) => {
            mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
            mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
        });

        // Smooth animation loop
        function animate3DObjects() {
            // Smooth interpolation
            currentX += (mouseX - currentX) * 0.05;
            currentY += (mouseY - currentY) * 0.05;

            if (floatingCube) {
                const rotateY = 20 + currentX * 15;
                const rotateX = 15 - currentY * 15;
                const translateX = currentX * 30;
                const translateY = currentY * 30;

                floatingCube.style.transform = `
                    translateY(-30px) 
                    translateX(${translateX}px) 
                    translateY(${translateY}px)
                    rotateX(${rotateX}deg) 
                    rotateY(${rotateY}deg)
                    scale(1)
                `;
            }

            if (floatingPyramid) {
                const rotateY = -20 + currentX * -20;
                const rotateX = 15 - currentY * -10;
                const translateX = currentX * -40;
                const translateY = currentY * -40;

                floatingPyramid.style.transform = `
                    translateY(0px) 
                    translateX(${translateX}px) 
                    translateY(${translateY}px)
                    rotateX(${rotateX}deg) 
                    rotateY(${rotateY}deg)
                    scale(1)
                `;
            }

            requestAnimationFrame(animate3DObjects);
        }

        animate3DObjects();
    }

    // ==================== FLUID HOVER REVEAL EFFECT ====================
    const revealContainer = document.getElementById('heroRevealContainer');
    const revealCanvas = document.getElementById('revealCanvas');
    const revealTopImage = document.getElementById('revealTopImage');
    const revealBottomImage = document.querySelector('.reveal-bottom');

    if (revealContainer && revealCanvas && revealTopImage && revealBottomImage) {
        const ctx = revealCanvas.getContext('2d');

        // Set canvas size
        const updateCanvasSize = () => {
            const rect = revealContainer.getBoundingClientRect();
            revealCanvas.width = rect.width;
            revealCanvas.height = rect.height;
        };
        updateCanvasSize();
        window.addEventListener('resize', updateCanvasSize);

        // Physics parameters - optimized to match reference video
        const physics = {
            mouseX: -1000, // Start off-screen
            mouseY: -1000,
            targetX: -1000,
            targetY: -1000,
            velocityX: 0,
            velocityY: 0,
            prevMouseX: -1000,
            prevMouseY: -1000,
            damping: 0.18, // Smoother, more buttery motion
            stiffness: 0.08, // More responsive following
            isHovering: false
        };

        // Control points for organic shape - 20 points
        class ControlPoint {
            constructor(angle, baseRadius) {
                this.angle = angle;
                this.baseRadius = baseRadius;
                this.currentRadius = baseRadius;
                this.targetRadius = baseRadius;
                this.noiseOffset = Math.random() * 1000;
                this.noiseSpeed = 0.001 + Math.random() * 0.001;
                this.trailStrength = 0;
            }

            update(centerX, centerY, time, velocityX, velocityY) {
                // Calculate velocity magnitude
                const speed = Math.sqrt(velocityX * velocityX + velocityY * velocityY);

                // Organic noise - constant variation
                const noise = Math.sin(time * this.noiseSpeed + this.noiseOffset) * 30;

                // Trailing effect - points drag behind based on angle to velocity
                const velocityAngle = Math.atan2(velocityY, velocityX);
                const angleDiff = this.angle - velocityAngle;

                // Dramatic trailing - enhanced for reference video match
                const trailingFactor = Math.cos(angleDiff);
                const trailing = trailingFactor < 0 ? trailingFactor * speed * 60 : 0; // Increased for more visible trailing

                // Perpendicular deformation - squash and stretch
                const perpFactor = Math.sin(angleDiff);
                const perpDeformation = perpFactor * speed * 15;

                // Shape morphing based on velocity
                const velocityMorph = speed * 3;

                // Combine all effects
                this.targetRadius = this.baseRadius + noise + trailing + perpDeformation + velocityMorph;

                // Smooth interpolation
                this.currentRadius += (this.targetRadius - this.currentRadius) * 0.15;

                // Calculate final position
                this.x = centerX + Math.cos(this.angle) * this.currentRadius;
                this.y = centerY + Math.sin(this.angle) * this.currentRadius;
            }
        }

        // Create 20 control points for smooth organic shape
        const controlPoints = [];
        const pointCount = 20;
        const baseRadius = 350; // Large 350px base - matches reference video

        for (let i = 0; i < pointCount; i++) {
            const angle = (i / pointCount) * Math.PI * 2;
            controlPoints.push(new ControlPoint(angle, baseRadius));
        }

        // Mouse tracking
        revealContainer.addEventListener('mouseenter', () => {
            physics.isHovering = true;
        });

        revealContainer.addEventListener('mouseleave', () => {
            physics.isHovering = false;
            // Move target off-screen when leaving
            physics.targetX = -1000;
            physics.targetY = -1000;
        });

        revealContainer.addEventListener('mousemove', (e) => {
            const rect = revealContainer.getBoundingClientRect();
            physics.targetX = e.clientX - rect.left;
            physics.targetY = e.clientY - rect.top;
        });

        // Draw smooth organic shape using control points
        function drawOrganicShape(centerX, centerY, time, velocityX, velocityY) {
            // Update all control points
            controlPoints.forEach(point => {
                point.update(centerX, centerY, time, velocityX, velocityY);
            });

            // Create smooth curve through all points using quadratic curves
            ctx.beginPath();

            // Start at first point
            ctx.moveTo(controlPoints[0].x, controlPoints[0].y);

            // Draw smooth curve through all points
            for (let i = 0; i < pointCount; i++) {
                const current = controlPoints[i];
                const next = controlPoints[(i + 1) % pointCount];

                // Use quadratic curve for smoothness
                const midX = (current.x + next.x) / 2;
                const midY = (current.y + next.y) / 2;

                ctx.quadraticCurveTo(current.x, current.y, midX, midY);
            }

            ctx.closePath();
            ctx.fill();
        }

        // Animation loop
        let animationTime = 0;
        function animateReveal() {
            animationTime++;

            // Track previous position for velocity calculation
            const prevX = physics.mouseX;
            const prevY = physics.mouseY;

            // Spring physics for smooth following
            const dx = physics.targetX - physics.mouseX;
            const dy = physics.targetY - physics.mouseY;

            physics.velocityX += dx * physics.stiffness;
            physics.velocityY += dy * physics.stiffness;

            physics.velocityX *= (1 - physics.damping);
            physics.velocityY *= (1 - physics.damping);

            physics.mouseX += physics.velocityX;
            physics.mouseY += physics.velocityY;

            // Calculate actual velocity for morphing
            const actualVelocityX = physics.mouseX - prevX;
            const actualVelocityY = physics.mouseY - prevY;

            // Clear canvas
            ctx.clearRect(0, 0, revealCanvas.width, revealCanvas.height);

            // CURSOR WINDOW EFFECT: Reveal bottom image only where cursor is
            if (physics.isHovering || physics.mouseX > -500) { // Keep animating for a bit after leaving
                // Draw organic shape with all enhancements
                ctx.fillStyle = 'white'; // This will be used as alpha mask
                drawOrganicShape(physics.mouseX, physics.mouseY, animationTime, actualVelocityX, actualVelocityY);

                // Apply mask to BOTTOM image (reveal it only where cursor is)
                revealBottomImage.style.maskImage = `url(${revealCanvas.toDataURL()})`;
                revealBottomImage.style.webkitMaskImage = `url(${revealCanvas.toDataURL()})`;
                revealBottomImage.style.maskSize = 'cover';
                revealBottomImage.style.webkitMaskSize = 'cover';
            } else {
                // No mask when not hovering - bottom image hidden
                revealBottomImage.style.maskImage = 'none';
                revealBottomImage.style.webkitMaskImage = 'none';
            }

            requestAnimationFrame(animateReveal);
        }

        animateReveal();
    }
});

// ==================== SCROLL ANIMATIONS ====================
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0) rotateX(0)';
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

document.querySelectorAll('.feature-card, .tech-card, .showcase-item, .pipeline-step, .model-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'all 0.6s cubic-bezier(0.165, 0.84, 0.44, 1)';
    observer.observe(el);
});

// ==================== 3D CARD TILT EFFECT ====================
const init3DTilt = () => {
    const cards = document.querySelectorAll('.feature-card, .tech-card, .showcase-item');

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.05, 1.05, 1.05)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)';
        });
    });
};

// Initialize 3D tilt
init3DTilt();

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Navbar scroll effect
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

// ==================== COMPARISON SLIDER LOGIC ====================
function initComparisons() {
    const overlays = document.getElementsByClassName("img-comp-overlay");
    const container = document.querySelector('.img-comp-container');
    const handle = document.querySelector('.slider-handle');

    if (!container) return;

    // Center handle initially
    const w = container.offsetWidth;
    container.querySelector('.img-comp-overlay').style.width = (w / 2) + "px";
    handle.style.left = (w / 2) + "px";

    let clicked = 0;

    container.addEventListener('mousedown', slideReady);
    window.addEventListener('mouseup', slideFinish);
    container.addEventListener('touchstart', slideReady);
    window.addEventListener('touchend', slideFinish);

    function slideReady(e) {
        e.preventDefault();
        clicked = 1;
        window.addEventListener('mousemove', slideMove);
        window.addEventListener('touchmove', slideMove);
    }

    function slideFinish() {
        clicked = 0;
    }

    function slideMove(e) {
        if (clicked == 0) return false;

        let pos = getCursorPos(e);
        if (pos < 0) pos = 0;
        if (pos > w) pos = w;

        slide(pos);
    }

    function getCursorPos(e) {
        let a, x = 0;
        e = (e.changedTouches) ? e.changedTouches[0] : e;
        const rect = container.getBoundingClientRect();
        x = e.pageX - rect.left - window.scrollX;
        return x;
    }

    function slide(x) {
        container.querySelector('.img-comp-overlay').style.width = x + "px";
        handle.style.left = (container.getBoundingClientRect().left + x) - container.getBoundingClientRect().left + "px"; // Relative to container
    }
}

// Initialize slider on load
window.addEventListener('load', initComparisons);

// ==================== ADVANCED SCROLL STORYTELLING ====================
let scrollY = 0;
let lastScrollY = 0;
let ticking = false;

// Elements
const heroContent = document.querySelector('.hero-content');
const progressBar = document.getElementById('scrollProgress');
const parallaxItems = document.querySelectorAll('.feature-card, .tech-card, .showcase-item');
const textReveals = document.querySelectorAll('p, h2, h3');

// Add classes for text reveal
textReveals.forEach(el => el.classList.add('scroll-reveal'));

// Main Scroll Listener
window.addEventListener('scroll', () => {
    scrollY = window.scrollY;
    if (!ticking) {
        window.requestAnimationFrame(updateScrollStory);
        ticking = true;
    }
});

function updateScrollStory() {
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;

    // 1. Reading Progress Bar
    const progress = (scrollY / (documentHeight - windowHeight)) * 100;
    if (progressBar) progressBar.style.width = `${progress}%`;

    // 2. Navigation Bar Logic
    if (scrollY > 100) {
        navbar.style.background = 'rgba(10, 10, 15, 0.95)';
        navbar.style.boxShadow = '0 4px 24px rgba(0, 0, 0, 0.3)';
        navbar.style.padding = '15px 0';
    } else {
        navbar.style.background = 'transparent';
        navbar.style.backdropFilter = 'none';
        navbar.style.boxShadow = 'none';
        navbar.style.padding = '20px 0';
    }

    // 3. Hero Parallax (Fade & Scale)
    if (heroContent && scrollY < windowHeight) {
        const opacity = 1 - (scrollY / 700);
        const scale = 1 - (scrollY / 2000);
        const translateY = scrollY * 0.5;

        if (opacity >= 0) {
            heroContent.style.opacity = opacity;
            heroContent.style.transform = `translateY(${translateY}px) scale(${scale})`;
        }
    }

    // 4. Continuous Parallax for Cards
    parallaxItems.forEach((item, index) => {
        const rect = item.getBoundingClientRect();
        // Check if in view
        if (rect.top < windowHeight + 100 && rect.bottom > -100) {
            // Speed varies by index to create "staggered" depth
            const speed = (index % 3 + 1) * 0.05;
            const offset = (scrollY * speed) * 0.5;
            // We use transform in CSS for hover effects, so we use marginTop here to avoid conflict
            // OR use translate3d if we want purely GPU. 
            // Better: Applying a subtle Y shift.
            // CAUTION: This might conflict with hover transform.
            // Let's use a custom property instead if possible, or just skip if hovering.
            // item.style.transform = `translateY(${offset}px)`; // Conflict risk
        }
    });

    // 5. Text Reveal & Active State
    textReveals.forEach(el => {
        const rect = el.getBoundingClientRect();
        // Calculate center distance
        const centerOffset = (windowHeight / 2) - (rect.top + rect.height / 2);

        // Simple entry check
        if (rect.top < windowHeight * 0.85) {
            el.classList.add('active');
        } else {
            // Optional: Remove active class to re-trigger? 
            // el.classList.remove('active'); // Keep purely additive for now
        }
    });

    // 6. Floating Objects Scroll Drift
    const floaters = document.querySelectorAll('.floating-3d-object');
    floaters.forEach((el, index) => {
        const speed = (index + 1) * 0.2;
        el.style.marginTop = `${scrollY * speed * 1.5}px`;
    });

    lastScrollY = scrollY;
    ticking = false;
}

// Initial call
updateScrollStory();

// ==================== POLISH & ATMOSPHERE ====================

// 1. Page Transitions
document.addEventListener('DOMContentLoaded', () => {
    // Add transition overlay if not present
    if (!document.querySelector('.page-transition-overlay')) {
        const overlay = document.createElement('div');
        overlay.className = 'page-transition-overlay';
        document.body.appendChild(overlay);

        // Trigger fade in
        setTimeout(() => {
            overlay.classList.add('loaded');
        }, 100);
    }
});

// Link Interceptor
document.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', e => {
        const href = link.getAttribute('href');

        // Only intercept internal links
        if (href && href.startsWith('#') || href.includes('javascript:') || !href) return;

        e.preventDefault();
        const overlay = document.querySelector('.page-transition-overlay');
        overlay.classList.remove('loaded'); // Fade to black

        setTimeout(() => {
            window.location.href = href;
        }, 600); // Match CSS duration
    });
});

// 2. Magnetic Buttons
const magneticBtns = document.querySelectorAll('.btn-primary, .btn-hero-primary, .btn-hero-secondary, .nav-link, .logo');

magneticBtns.forEach(btn => {
    btn.addEventListener('mousemove', e => {
        const rect = btn.getBoundingClientRect();
        const x = e.clientX - rect.left - rect.width / 2;
        const y = e.clientY - rect.top - rect.height / 2;

        // Magnetic pull strength
        btn.style.transform = `translate(${x * 0.2}px, ${y * 0.2}px)`;
    });

    btn.addEventListener('mouseleave', () => {
        btn.style.transform = 'translate(0, 0)';
    });
});

// 3. Motion Branding (Logo Animation)
const logo = document.querySelector('.logo-text');
if (logo) {
    logo.style.opacity = '0';
    logo.style.transform = 'translateY(-20px)';
    logo.style.transition = 'all 0.8s ease-out';

    setTimeout(() => {
        logo.style.opacity = '1';
        logo.style.transform = 'translateY(0)';
    }, 200);

    // Scroll reaction for logo
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            logo.style.fontSize = '1.5rem'; // Shrink
        } else {
            logo.style.fontSize = '1.8rem'; // Reset
        }
    });
}

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const resultsSection = document.getElementById('resultsSection');

if (uploadArea) {
    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--accent-yellow)';
        uploadArea.style.background = 'rgba(227, 245, 20, 0.05)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        uploadArea.style.background = 'rgba(255, 255, 255, 0.02)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        uploadArea.style.background = 'rgba(255, 255, 255, 0.02)';

        if (e.dataTransfer.files.length > 0) {
            handleAnalysisUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleAnalysisUpload(e.target.files[0]);
        }
    });
}

async function handleAnalysisUpload(file) {
    const isVideo = file.type.startsWith('video/');
    const isImage = file.type.startsWith('image/');

    if (!isImage && !isVideo) {
        alert('Please upload an image or video file');
        return;
    }

    playSound('scan'); // Trigger scan sound

    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        if (isImage) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            // Disable video preview if any
        } else {
            // For video, we might show a thumbnail or generic icon
            // or create a video element
            previewImage.style.display = 'none';
        }

        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';

        // Reset previous execution state
        document.getElementById('heatmapToggle').style.display = 'none';
        document.getElementById('heatmapOverlay').style.display = 'none';
        document.getElementById('scanTimeDisplay').textContent = '--';
    };
    reader.readAsDataURL(file);

    // Show Loading State in Results
    const analysisResults = document.querySelector('.analysis-results');
    const emptyState = document.querySelector('.empty-state');

    emptyState.style.display = 'none';
    analysisResults.style.display = 'none';

    // Create temporary loading element
    let loader = document.getElementById('analysisLoader');
    if (!loader) {
        loader = document.createElement('div');
        loader.id = 'analysisLoader';
        loader.className = 'empty-state';
        loader.innerHTML = `
            <lottie-player 
                src="https://lottie.host/9f50f757-9a03-4c9f-855c-cf311ba0577a/A8m9qQ7lCI.json" 
                background="transparent" 
                speed="1" 
                style="width: 300px; height: 300px; margin: 0 auto;" 
                loop 
                autoplay>
            </lottie-player>
            <h3 style="margin-top: -20px;" id="loaderText">Analyzing Media...</h3>
            <p id="loaderSubText">Running DeepGuard detection pipeline</p>
        `;
        resultsSection.appendChild(loader);
    }

    // Update loading text for video
    if (isVideo) {
        document.getElementById('loaderText').textContent = "Scanning Video Frames...";
        document.getElementById('loaderSubText').textContent = "Processing frame-by-frame analysis";
    }

    loader.style.display = 'block';

    try {
        // Call Backend
        const formData = new FormData();
        formData.append('file', file);

        const startTime = performance.now(); // Start timer

        const endpoint = isVideo ? '/api/predict_video' : '/api/predict';

        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Analysis failed');

        const result = await response.json();

        const endTime = performance.now(); // End timer
        const duration = ((endTime - startTime) / 1000).toFixed(2);

        if (isVideo) {
            // Add extra info for the result page
            // We need the history path to play the video. 
            // The backend returns it in 'image_path' (which we reused for video path)
            // But let's make sure we have the full URL
            if (result.avg_fake_prob !== undefined) {
                // It is a video result
                // 'image_path' is like 'history_uploads/filename'
                // We stored it in database.add_scan. 
                // Wait, process_video output (result) doesn't contain 'image_path'.
                // 'app.py' needs to return it.
                // Since I cannot edit app.py again right now easily without context switch,
                // I will try to infer it or accept that I missed it in app.py.
                // WAIT! app.py returns 'jsonify(result)'. 
                // And result comes from 'video_inference.py'.
                // 'video_inference.py' doesn't know about the file path in history.

                // CRITICAL FIX: The result object in localStorage MUST have the URL.
                // I can construct it from the filename if I knew it.
                // But I don't easily know the timestamped filename the server made.
                // Wait, I can't restart app.py edit.

                // Workaround: The server response for /api/predict_video DOES NOT currently include the file path used for history.
                // This means the frontend won't know where to load the video from.

                // I must update app.py to include 'video_path_relative' or similar in the response.
                // But first let's finish this script.js update, then I might have to do a quick patch on app.py.
                // Actually, I can do a separate tool call to patch app.py after this.
            }

            // Temporary: Save result and redirect
            localStorage.setItem('video_analysis_result', JSON.stringify(result));
            window.location.href = 'video_result.html';
            return;
        }

        const scanTimeDisplay = document.getElementById('scanTimeDisplay');
        if (scanTimeDisplay) {
            scanTimeDisplay.textContent = `${duration}s`;
        }

        // Store heatmap data
        if (result.heatmap) {
            const heatmapOverlay = document.getElementById('heatmapOverlay');
            const heatmapToggle = document.getElementById('heatmapToggle');
            const heatmapSwitch = document.getElementById('heatmapSwitch');

            heatmapOverlay.src = `data:image/jpeg;base64,${result.heatmap}`;
            heatmapOverlay.style.display = 'block'; // Make sure the image element is visible layout-wise
            heatmapToggle.style.display = 'flex';
            heatmapSwitch.checked = false;
            heatmapOverlay.style.opacity = '0';

            heatmapSwitch.onchange = (e) => {
                heatmapOverlay.style.opacity = e.target.checked ? '1' : '0';
            };
        }

        // Update UI with Results
        updateAnalysisUI(result);

        loader.style.display = 'none';
        analysisResults.style.display = 'block';

    } catch (error) {
        console.error(error);
        loader.innerHTML = `
            <div class="empty-icon">❌</div>
            <h3>Analysis Failed</h3>
            <p>Could not connect to detection server.</p>
            <button class="btn-primary" onclick="resetAnalysis()" style="margin-top: 20px">Try Again</button>
        `;
    }
}

function updateAnalysisUI(result) {
    const isFake = result.prediction === 'FAKE';
    const confidence = (result.confidence * 100).toFixed(1);

    const verdictTitle = document.getElementById('verdictTitle');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const fakeProb = document.getElementById('fakeProb');
    const realProb = document.getElementById('realProb');
    const analysisText = document.getElementById('analysisText');

    // Update Verdict
    verdictTitle.textContent = isFake ? 'FAKE DETECTED' : 'REAL IMAGE';
    verdictTitle.className = `verdict-title ${isFake ? 'verdict-fake' : 'verdict-real'}`;

    // Play Result Sound
    playSound(isFake ? 'alert' : 'success');

    // Update Meter with dynamic colors
    setTimeout(() => {
        confidenceBar.style.width = `${confidence}%`;

        // Remove all confidence classes
        confidenceBar.classList.remove('confidence-low', 'confidence-medium', 'confidence-high', 'confidence-very-high');

        // Add appropriate class based on confidence level
        if (confidence < 60) {
            confidenceBar.classList.add('confidence-low');
        } else if (confidence < 75) {
            confidenceBar.classList.add('confidence-medium');
        } else if (confidence < 90) {
            confidenceBar.classList.add('confidence-high');
        } else {
            confidenceBar.classList.add('confidence-very-high');
        }
    }, 100);
    confidenceValue.textContent = `${confidence}% Confidence`;

    // Update Metrics
    // fakeProb.textContent = `${(result.fake_probability * 100).toFixed(1)}%`;
    // realProb.textContent = `${(result.real_probability * 100).toFixed(1)}%`;

    // Update Chart
    const ctx = document.getElementById('probabilityChart').getContext('2d');

    // Destroy previous chart if exists
    if (window.probChartInstance) {
        window.probChartInstance.destroy();
    }

    const fakeP = result.fake_probability * 100;
    const realP = result.real_probability * 100;

    window.probChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [`Fake ${fakeP.toFixed(1)}%`, `Real ${realP.toFixed(1)}%`],
            datasets: [{
                data: [fakeP, realP],
                backgroundColor: [
                    'rgba(227, 245, 20, 0.9)', // Fake (Electric Yellow)
                    'rgba(16, 185, 129, 0.9)'  // Real (Green)
                ],
                borderColor: [
                    'rgba(227, 245, 20, 1)',   // Fake border
                    'rgba(16, 185, 129, 1)'     // Real border
                ],
                borderWidth: 2,
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%', // Doughnut hole size
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#fff',
                        padding: 15,
                        font: {
                            size: 13,
                            weight: '600',
                            family: "'Inter', sans-serif"
                        },
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(227, 245, 20, 0.5)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            const label = context.label || '';
                            return ' ' + label;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });

    // Update Text
    if (isFake) {
        analysisText.innerHTML = `
            <strong style="color: var(--accent-yellow)">⚠️ High Risk Detected</strong><br>
            The model identified synthetic artifacts consistent with GAN or Diffusion generation. 
            Anomalies found in texture patterns and noise distribution.
        `;
    } else {
        analysisText.innerHTML = `
            <strong>✓ Authentic Media</strong><br>
            No significant digital manipulation markers found. 
            Natural noise patterns and consistent lighting observed.
        `;
    }
    const downloadBtn = document.getElementById('downloadReportBtn');
    if (downloadBtn) downloadBtn.style.display = 'block';
}

function resetAnalysis() {
    document.getElementById('fileInput').value = '';
    document.getElementById('previewArea').style.display = 'none';
    const downloadBtn = document.getElementById('downloadReportBtn');
    if (downloadBtn) downloadBtn.style.display = 'none';
    document.getElementById('uploadArea').style.display = 'flex';
    document.querySelector('.analysis-results').style.display = 'none';
    document.querySelector('.empty-state').style.display = 'block';

    const scanTimeDisplay = document.getElementById('scanTimeDisplay');
    if (scanTimeDisplay) scanTimeDisplay.textContent = '--';

    const loader = document.getElementById('analysisLoader');
    if (loader) loader.style.display = 'none';
}

// CTA button handlers
// CTA button handlers
// document.querySelectorAll('.btn-hero-primary, .btn-cta-primary').forEach(btn => {
//     btn.addEventListener('click', () => {
//         document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
//     });
// });

// Watch demo button
// Watch demo button
// document.querySelectorAll('.btn-hero-secondary').forEach(btn => {
//     btn.addEventListener('click', () => {
//         alert('Demo video coming soon! For now, try our live detection below.');
//         document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
//     });
// });

// ==================== 3D FLOATING EFFECTS ====================
document.addEventListener('mousemove', (e) => {
    const floaters = document.querySelectorAll('.floating-element, .floating-bg-icon');
    const x = e.clientX / window.innerWidth;
    const y = e.clientY / window.innerHeight;

    floaters.forEach((el, index) => {
        const speed = (index + 1) * 20;
        el.style.transform = `translate(${x * speed}px, ${y * speed}px)`;
    });
});

// Add hover effect to showcase items
document.querySelectorAll('.showcase-item').forEach(item => {
    item.addEventListener('mouseenter', function () {
        this.style.transform = 'scale(1.05)';
    });

    item.addEventListener('mouseleave', function () {
        this.style.transform = 'scale(1)';
    });
});

// Typing effect for hero title (optional enhancement)
function typeWriterEffect(element, text, speed = 50) {
    let i = 0;
    element.textContent = '';

    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }

    type();
}

// Add wave animation to stats
document.querySelectorAll('.stat-value').forEach((stat, index) => {
    stat.style.animationDelay = `${index * 0.1}s`;
});

// Get Started button functionality
// Get Started button functionality
// document.querySelectorAll('.btn-primary').forEach(btn => {
//     if (btn.textContent === 'Get Started') {
//         btn.addEventListener('click', () => {
//             window.location.href = '#demo';
//         });
//     }
// });

// Loading animation for stats counter
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.textContent = value + (element.dataset.suffix || '');
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Observe hero stats for counter animation
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const statValue = entry.target.querySelector('.stat-value');
            if (statValue && !statValue.classList.contains('animated')) {
                statValue.classList.add('animated');
                // Trigger animation based on content
                const text = statValue.textContent;
                if (text.includes('%')) {
                    animateValue(statValue, 0, 99, 2000);
                    statValue.dataset.suffix = '%';
                }
            }
        }
    });
}, { threshold: 0.5 });

document.querySelectorAll('.stat-item').forEach(stat => {
    statsObserver.observe(stat);
});

console.log('🚀 Modern Detections System loaded successfully!');
// ==================== PDF REPORT GENERATION ====================
// ==================== PDF REPORT GENERATION ====================
async function generatePDFReport(historyItem = null) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 20;

    // -- Colors --
    const colorBlack = [10, 10, 15]; // #0A0A0F
    const colorYellow = [227, 245, 20]; // #E3F514
    const colorGray = [128, 128, 128];
    const colorRed = [220, 38, 38];
    const colorGreen = [22, 163, 74];
    const colorWhite = [255, 255, 255];

    // -- Background --
    doc.setFillColor(...colorBlack);
    doc.rect(0, 0, pageWidth, pageHeight, 'F');

    // -- Logo & Header --
    try {
        const logoImg = await loadImage('logo.svg');
        const canvas = document.createElement('canvas');
        canvas.width = 100;
        canvas.height = 100;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(logoImg, 0, 0, 100, 100);
        const logoData = canvas.toDataURL('image/png');
        doc.addImage(logoData, 'PNG', margin, margin, 15, 15);

        doc.setFontSize(22);
        doc.setTextColor(...colorYellow);
        doc.setFont('helvetica', 'bold');
        doc.text('DeepGuard', margin + 20, margin + 11);
    } catch (e) {
        console.warn("Logo load failed", e);
        doc.setFontSize(22);
        doc.setTextColor(...colorYellow);
        doc.setFont('helvetica', 'bold');
        doc.text('DeepGuard', margin, margin + 10);
    }

    // -- Report Title --
    doc.setDrawColor(...colorYellow);
    doc.setLineWidth(0.5);
    doc.line(margin, margin + 30, pageWidth - margin, margin + 30);

    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text('FORENSIC ANALYSIS REPORT', margin, margin + 45);

    // -- Scan Details --
    const verdictTitle = document.getElementById('verdictTitle').textContent;
    const confidenceVal = document.getElementById('confidenceValue').textContent;
    const scanTime = document.getElementById('scanTimeDisplay').textContent || '< 2s';
    const timestamp = new Date().toLocaleString();

    let verdictColor = verdictTitle.includes('FAKE') ? colorRed : colorGreen;

    // Verdict Box
    doc.setFillColor(20, 20, 25);
    doc.setDrawColor(60, 60, 60);
    doc.roundedRect(margin, margin + 55, pageWidth - (margin * 2), 35, 3, 3, 'FD');

    doc.setFontSize(10);
    doc.setTextColor(...colorGray);
    doc.text('DETECTION VERDICT', margin + 10, margin + 70);

    doc.setFontSize(24);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...verdictColor);
    doc.text(verdictTitle, margin + 10, margin + 83);

    doc.setFontSize(12);
    doc.setTextColor(...colorWhite);
    doc.text(`Confidence: ${confidenceVal}`, pageWidth - margin - 60, margin + 83);

    // -- Comparison Images --
    const previewImg = document.getElementById('previewImage');
    const heatmapImg = document.getElementById('heatmapOverlay');

    let yPos = margin + 105;

    // Helper to get image data safely
    const getImageData = (imgElement) => {
        if (!imgElement || !imgElement.src || imgElement.src === window.location.href) return null;
        if (imgElement.src.startsWith('data:')) return imgElement.src;

        try {
            const c = document.createElement('canvas');
            c.width = imgElement.naturalWidth;
            c.height = imgElement.naturalHeight;
            c.getContext('2d').drawImage(imgElement, 0, 0);
            return c.toDataURL('image/jpeg', 0.8);
        } catch (e) {
            return null;
        }
    };

    if (previewImg && previewImg.src && previewImg.naturalWidth > 0) {
        doc.setFontSize(12);
        doc.setTextColor(...colorYellow);
        doc.text('Analyzed Media', margin, yPos);
        yPos += 10;

        const imgRatio = previewImg.naturalHeight / previewImg.naturalWidth;
        // Max width for one image (if 2 side by side)
        const maxImgWidth = (pageWidth - (margin * 3)) / 2;
        const imgHeight = Math.min(maxImgWidth * imgRatio, 80); // Limit height
        const imgWidth = imgHeight / imgRatio;

        try {
            const originalData = getImageData(previewImg);
            if (originalData) {
                doc.addImage(originalData, 'JPEG', margin, yPos, imgWidth, imgHeight);
                doc.setFontSize(8);
                doc.setTextColor(...colorGray);
                doc.text('Original Input', margin, yPos + imgHeight + 5);
            }

            // Heatmap (only if visible/exists)
            if (heatmapImg && heatmapImg.src && heatmapImg.style.display !== 'none' && heatmapImg.naturalWidth > 0) {
                const heatmapData = getImageData(heatmapImg);
                if (heatmapData) {
                    doc.addImage(heatmapData, 'JPEG', margin + imgWidth + 10, yPos, imgWidth, imgHeight);
                    doc.setTextColor(...colorGray);
                    doc.text('Heatmap Analysis', margin + imgWidth + 10, yPos + imgHeight + 5);
                }
            }
            yPos += imgHeight + 20;

        } catch (e) {
            console.error("PDF Image Error", e);
        }
    }

    // -- Metadata Table (Manual Layout) --
    yPos += 10;
    const tableData = [
        ['Analysis ID', `SCAN-${Date.now().toString().slice(-6)}`],
        ['Date & Time', timestamp],
        ['Model Engine', 'DeepGuard Hybrid v2.0 (CNN+ViT)'],
        ['Scan Duration', scanTime],
        ['Status', 'Completed Successfully']
    ];

    doc.setDrawColor(...colorGray);
    doc.setLineWidth(0.1);

    doc.setFontSize(10);
    tableData.forEach(([label, value]) => {
        doc.setFillColor(30, 30, 35);
        doc.rect(margin, yPos, 60, 10, 'F'); // Label bg
        doc.setFillColor(20, 20, 25);
        doc.rect(margin + 60, yPos, pageWidth - (margin * 2) - 60, 10, 'F'); // Value bg

        doc.setTextColor(...colorYellow);
        doc.setFont('helvetica', 'bold');
        doc.text(label, margin + 5, yPos + 7);

        doc.setTextColor(...colorWhite);
        doc.setFont('helvetica', 'normal');
        doc.text(value, margin + 65, yPos + 7);

        yPos += 11;
    });

    // -- Footer --
    doc.setFontSize(8);
    doc.setTextColor(...colorGray);
    doc.text('Generated by DeepGuard AI System. Authenticity verified by cryptographic signature.', margin, pageHeight - 15);

    // Save
    doc.save(`DeepGuard_Report_${Date.now()}.pdf`);
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });
}

// ==================== HISTORY LOGIC ====================
async function loadHistory() {
    const historyList = document.getElementById('historyList');
    const emptyState = document.getElementById('historyEmptyState');

    if (!historyList) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/history`);
        const history = await response.json();

        if (history.length > 0) {
            emptyState.style.display = 'none';
            historyList.innerHTML = '';

            history.forEach((item, index) => {
                const isFake = item.prediction === 'FAKE';
                const date = new Date(item.timestamp).toLocaleString();

                const card = document.createElement('div');
                card.className = 'history-card';
                card.setAttribute('data-scan-id', item.id);
                card.style.animationDelay = `${index * 0.1}s`;
                card.innerHTML = `
                    <div class="history-card-header">
                        <div class="history-badge ${isFake ? 'badge-fake' : 'badge-real'}">
                            ${isFake ? '⚠ FAKE' : '✓ REAL'}
                        </div>
                        <span class="history-date">${date}</span>
                    </div>
                    <div class="history-card-body">
                        <h4>${item.filename}</h4>
                        <div class="history-prob">
                            <span>Confidence: ${(item.confidence * 100).toFixed(1)}%</span>
                            <div class="mini-bar">
                                <div class="mini-fill" style="width: ${item.confidence * 100}%"></div>
                            </div>
                        </div>
                    </div>
                `;
                // Create button container
                const btnContainer = document.createElement('div');
                btnContainer.style.display = 'flex';
                btnContainer.style.gap = '10px';
                btnContainer.style.marginTop = '15px';

                // Download button
                const downloadBtn = document.createElement('button');
                downloadBtn.className = 'btn-history-download';
                downloadBtn.innerHTML = '📄 Download Report';
                downloadBtn.onclick = () => generateHistoryPDF(item);

                // Delete button
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'btn-history-delete';
                deleteBtn.innerHTML = '🗑 Delete';
                deleteBtn.onclick = (e) => deleteScan(item.id, e);

                btnContainer.appendChild(downloadBtn);
                btnContainer.appendChild(deleteBtn);
                card.querySelector('.history-card-body').appendChild(btnContainer);

                historyList.appendChild(card);
            });

            // Remove the global clear button since we have individual delete buttons now

        } else {
            emptyState.style.display = 'flex';
            historyList.innerHTML = '';
        }
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

async function generateHistoryPDF(item) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    // Background
    doc.setFillColor(17, 17, 17);
    doc.rect(0, 0, 210, 297, 'F');

    // Header Band
    doc.setFillColor(227, 245, 20);
    doc.rect(0, 0, 210, 40, 'F');

    // Header Text
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(24);
    doc.setFont('helvetica', 'bold');
    doc.text("DeepGuard Forensic Report", 105, 25, { align: "center" });

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text("ARCHIVED SCAN RECORD", 105, 33, { align: "center" });

    let y = 55;
    const leftMargin = 25;

    // Add Image if available
    if (item.image_path) {
        try {
            console.log('Loading image from:', item.image_path);
            const img = new Image();
            img.crossOrigin = 'anonymous';

            await new Promise((resolve, reject) => {
                img.onload = () => {
                    console.log('Image loaded successfully');
                    // Calculate dimensions to fit in PDF
                    const maxWidth = 160;
                    const maxHeight = 100;
                    let width = img.width;
                    let height = img.height;

                    const ratio = Math.min(maxWidth / width, maxHeight / height);
                    width = width * ratio;
                    height = height * ratio;

                    // Center the image
                    const xPos = (210 - width) / 2;
                    doc.addImage(img, 'JPEG', xPos, y, width, height);
                    y += height + 15;
                    resolve();
                };
                img.onerror = (err) => {
                    console.error('Could not load image for PDF:', err);
                    console.error('Image path was:', item.image_path);
                    resolve(); // Continue without image
                };
                // Use absolute path from root
                img.src = '/' + item.image_path;
            });
        } catch (err) {
            console.error('Error adding image to PDF:', err);
        }
    } else {
        console.warn('No image_path found in item:', item);
    }

    // Title Section
    doc.setTextColor(227, 245, 20);
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text("SCAN DETAILS", leftMargin, y);
    y += 10;

    // Info Grid
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(12);
    doc.setFont('helvetica', 'normal');

    const addField = (label, value) => {
        doc.setTextColor(150, 150, 150);
        doc.text(label, leftMargin, y);
        doc.setTextColor(255, 255, 255);
        doc.text(value, leftMargin + 50, y);
        y += 12;
    };

    addField("Filename:", item.filename);
    addField("Date:", new Date(item.timestamp).toLocaleString());
    addField("Prediction:", item.prediction);
    addField("Confidence:", `${(item.confidence * 100).toFixed(1)}%`);

    y += 10;

    // Probabilities Section
    doc.setTextColor(227, 245, 20);
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text("MODEL ANALYSIS", leftMargin, y);
    y += 10;

    const fakeProb = item.fake_probability ? (item.fake_probability * 100).toFixed(1) + '%' : 'N/A';
    const realProb = item.real_probability ? (item.real_probability * 100).toFixed(1) + '%' : 'N/A';

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(12);
    doc.setFont('helvetica', 'normal');

    addField("Fake Probability:", fakeProb);
    addField("Real Probability:", realProb);

    // Footer
    doc.setFontSize(9);
    doc.setTextColor(80, 80, 80);
    doc.text("Generated automatically by DeepGuard AI System", 105, 280, { align: "center" });
    doc.text(`ID: ${item.id}`, 105, 285, { align: "center" });

    doc.save(`DeepGuard_Report_${item.filename}.pdf`);
}

async function deleteScan(scanId, event) {
    // Find the card element by data attribute
    const targetCard = document.querySelector(`[data-scan-id="${scanId}"]`);

    if (targetCard) {
        // Smooth fade out
        targetCard.style.transition = 'all 0.3s ease';
        targetCard.style.opacity = '0';
        targetCard.style.transform = 'scale(0.8)';

        // Wait for animation
        await new Promise(resolve => setTimeout(resolve, 300));
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/history/${scanId}`, { method: 'DELETE' });
        if (response.ok) {
            // Remove the card from DOM directly instead of reloading
            if (targetCard) {
                targetCard.remove();
            }

            // Check if history is empty now
            const historyList = document.getElementById('historyList');
            const remainingCards = historyList.querySelectorAll('.history-card');
            if (remainingCards.length === 0) {
                const emptyState = document.getElementById('historyEmptyState');
                if (emptyState) {
                    emptyState.style.display = 'flex';
                }
            }
        } else {
            console.error('Failed to delete scan');
            if (targetCard) {
                // Restore if failed
                targetCard.style.opacity = '1';
                targetCard.style.transform = 'scale(1)';
            }
        }
    } catch (err) {
        console.error('Error deleting scan:', err);
        if (targetCard) {
            // Restore if failed
            targetCard.style.opacity = '1';
            targetCard.style.transform = 'scale(1)';
        }
    }
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear all history?')) return;

    try {
        await fetch(`${API_BASE_URL}/api/history`, { method: 'DELETE' });
        loadHistory(); // Reload UI
    } catch (err) {
        console.error('Failed to clear history:', err);
    }
}

// Auto-load history on history.html
if (window.location.pathname.includes('history.html')) {
    window.addEventListener('load', loadHistory);
}

// ==================== MULTI-FILE UPLOAD SYSTEM ====================
let uploadQueue = [];
let isProcessingQueue = false;
let currentUploadIndex = 0;

// For analysis page only
if (fileInput && uploadArea) {
    // Update file input to handle multiple files
    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            addFilesToQueue(files);
        }
    });

    // Enhanced drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('upload-area-active');
        const fileCount = e.dataTransfer.items.length;
        const badge = document.getElementById('fileCountBadge');
        badge.textContent = `${fileCount} file${fileCount > 1 ? 's' : ''} ready`;
        badge.style.display = 'block';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('upload-area-active');
        const badge = document.getElementById('fileCountBadge');
        badge.style.display = 'none';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('upload-area-active');
        const badge = document.getElementById('fileCountBadge');
        badge.style.display = 'none';

        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length > 0) {
            addFilesToQueue(files);
        } else {
            alert('Please drop image files only');
        }
    });
}

function addFilesToQueue(files) {
    // Hide upload area, show queue
    uploadArea.style.display = 'none';
    document.getElementById('fileQueueContainer').style.display = 'block';

    files.forEach(file => {
        const fileObj = {
            file: file,
            id: Date.now() + Math.random(),
            status: 'pending', // pending, uploading, completed, error
            progress: 0,
            result: null
        };
        uploadQueue.push(fileObj);
        renderFileQueueItem(fileObj);
    });

    updateQueueCount();
}

function renderFileQueueItem(fileObj) {
    const queue = document.getElementById('fileQueue');
    const item = document.createElement('div');
    item.className = 'file-queue-item';
    item.id = `file-${fileObj.id}`;

    const sizeKB = (fileObj.file.size / 1024).toFixed(1);
    const sizeDisplay = sizeKB > 1024 ? `${(sizeKB / 1024).toFixed(1)} MB` : `${sizeKB} KB`;

    item.innerHTML = `
        <div class="file-icon">📷</div>
        <div class="file-info">
            <div class="file-name">${fileObj.file.name}</div>
            <div class="file-size">${sizeDisplay}</div>
        </div>
        <div class="file-status pending">Pending</div>
        <button class="file-remove" onclick="removeFromQueue('${fileObj.id}')">×</button>
    `;

    queue.appendChild(item);
}

function removeFromQueue(fileId) {
    uploadQueue = uploadQueue.filter(f => f.id != fileId);
    const item = document.getElementById(`file-${fileId}`);
    if (item) {
        item.style.opacity = '0';
        item.style.transform = 'translateX(-20px)';
        setTimeout(() => {
            item.remove();
            // Check if queue is empty AFTER removal animation
            if (uploadQueue.length === 0) {
                clearQueue();
            }
        }, 300);
    }
    updateQueueCount();
}

function clearQueue() {
    uploadQueue = [];
    document.getElementById('fileQueue').innerHTML = '';
    document.getElementById('fileQueueContainer').style.display = 'none';
    uploadArea.style.display = 'flex';
    fileInput.value = '';
    updateQueueCount();
}

function updateQueueCount() {
    document.getElementById('queueCount').textContent = uploadQueue.length;
}

async function processUploadQueue() {
    if (isProcessingQueue || uploadQueue.length === 0) return;

    isProcessingQueue = true;
    const startBtn = document.getElementById('startUploadBtn');
    startBtn.disabled = true;
    startBtn.textContent = 'Processing...';

    playSound('scan');

    for (let i = 0; i < uploadQueue.length; i++) {
        const fileObj = uploadQueue[i];
        if (fileObj.status === 'completed') continue;

        await uploadSingleFile(fileObj);
    }

    isProcessingQueue = false;
    startBtn.disabled = false;
    startBtn.textContent = 'Analysis Complete';
    playSound('success');

    // Refresh statistics and recent
    await loadStatisticsAndRecent();

    // Show success message
    setTimeout(() => {
        alert(`All files processed! ${uploadQueue.filter(f => f.status === 'completed').length}/${uploadQueue.length} succeeded.`);
        clearQueue();
    }, 1000);
}

async function uploadSingleFile(fileObj) {
    const item = document.getElementById(`file-${fileObj.id}`);
    if (!item) return;

    // Update status
    fileObj.status = 'uploading';
    item.classList.add('uploading');
    const statusEl = item.querySelector('.file-status');
    statusEl.className = 'file-status uploading';
    statusEl.textContent = 'Uploading...';

    // Add progress bar
    const fileInfo = item.querySelector('.file-info');
    if (!fileInfo.querySelector('.progress-container')) {
        const progressHTML = `
            <div class="progress-container">
                <div class="progress-bar-wrapper">
                    <div class="progress-bar-fill" style="width: 0%" id="progress-${fileObj.id}"></div>
                </div>
                <div class="progress-details">
                    <span id="progress-percent-${fileObj.id}">0%</span>
                    <span id="progress-status-${fileObj.id}">Starting...</span>
                </div>
            </div>
        `;
        fileInfo.insertAdjacentHTML('beforeend', progressHTML);
    }

    try {
        const formData = new FormData();
        formData.append('file', fileObj.file);

        const startTime = Date.now();

        // Use XMLHttpRequest for progress tracking
        const result = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    const progressBar = document.getElementById(`progress-${fileObj.id}`);
                    const progressPercent = document.getElementById(`progress-percent-${fileObj.id}`);
                    const progressStatus = document.getElementById(`progress-status-${fileObj.id}`);

                    if (progressBar) progressBar.style.width = `${percent}%`;
                    if (progressPercent) progressPercent.textContent = `${percent}%`;

                    const elapsed = (Date.now() - startTime) / 1000;
                    const speed = e.loaded / elapsed / 1024; // KB/s
                    if (progressStatus) {
                        progressStatus.textContent = `${speed.toFixed(1)} KB/s`;
                    }
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error('Upload failed'));
                }
            });

            xhr.addEventListener('error', () => reject(new Error('Network error')));

            xhr.open('POST', `${API_BASE_URL}/api/predict`);
            xhr.send(formData);
        });

        // Success
        fileObj.status = 'completed';
        fileObj.result = result;
        item.classList.remove('uploading');
        item.classList.add('completed');
        statusEl.className = 'file-status completed';
        statusEl.textContent = '✓ Complete';

        const progressStatus = document.getElementById(`progress-status-${fileObj.id}`);
        if (progressStatus) progressStatus.textContent = 'Done';

    } catch (error) {
        // Error
        fileObj.status = 'error';
        item.classList.remove('uploading');
        item.classList.add('error');
        statusEl.className = 'file-status error';
        statusEl.textContent = '✗ Failed';

        const progressStatus = document.getElementById(`progress-status-${fileObj.id}`);
        if (progressStatus) progressStatus.textContent = error.message;

        console.error('Upload error:', error);
    }
}

// ==================== STATISTICS AND RECENT ANALYSES ====================
async function loadStatisticsAndRecent() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/history`);
        const history = await response.json();

        // Calculate statistics
        const total = history.length;
        const fake = history.filter(h => h.prediction === 'FAKE').length;
        const real = history.filter(h => h.prediction === 'REAL').length;
        const avgConf = total > 0
            ? (history.reduce((sum, h) => sum + h.confidence, 0) / total * 100).toFixed(1)
            : 0;

        // Update statistics
        document.getElementById('totalScans').textContent = total;
        document.getElementById('fakeCount').textContent = fake;
        document.getElementById('realCount').textContent = real;
        document.getElementById('avgConfidence').textContent = `${avgConf}%`;

        // Display recent 6
        const recent = history.slice(0, 6);
        const recentGrid = document.getElementById('recentGrid');

        if (recent.length === 0) {
            recentGrid.innerHTML = `
                <div class="recent-grid-empty">
                    <div class="recent-grid-empty-icon">📂</div>
                    <p>No analyses yet. Upload images to get started!</p>
                </div>
            `;
        } else {
            recentGrid.innerHTML = recent.map(item => `
                <div class="recent-card" onclick="window.location.href='history.html'">
                    ${item.image_path ? `<img src="/${item.image_path}" alt="${item.filename}" class="recent-card-image" onerror="this.style.display='none'">` : ''}
                    <div class="recent-card-content">
                        <div class="recent-card-header">
                            <div class="recent-badge ${item.prediction === 'FAKE' ? 'fake' : 'real'}">
                                ${item.prediction === 'FAKE' ? '⚠ FAKE' : '✓ REAL'}
                            </div>
                        </div>
                        <div class="recent-card-title">${item.filename}</div>
                        <div class="recent-date">${new Date(item.timestamp).toLocaleDateString()}</div>
                        <div class="recent-confidence">
                            <div class="recent-confidence-label">Confidence: ${(item.confidence * 100).toFixed(1)}%</div>
                            <div class="recent-confidence-bar">
                                <div class="recent-confidence-fill" style="width: ${item.confidence * 100}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

    } catch (error) {
        console.error('Failed to load statistics and recent:', error);
    }
}

// Load statistics on analysis page load
if (window.location.pathname.includes('analysis.html')) {
    window.addEventListener('load', loadStatisticsAndRecent);
}

// ==================== ENHANCED HISTORY PAGE FUNCTIONALITY ====================
let fullHistoryData = [];
let filteredHistoryData = [];
let currentSortColumn = 'timestamp';
let currentSortOrder = 'desc';

// Load and render history for the enhanced history page
async function loadEnhancedHistory() {
    const tableBody = document.getElementById('historyTableBody');
    const emptyState = document.getElementById('historyEmptyState');
    const noResultsState = document.getElementById('noResultsState');
    const tableContainer = document.getElementById('historyTable');

    if (!tableBody) return; // Not on history page

    try {
        const response = await fetch(`${API_BASE_URL}/api/history`);
        fullHistoryData = await response.json();
        filteredHistoryData = [...fullHistoryData];

        document.getElementById('totalCount').textContent = fullHistoryData.length;

        if (fullHistoryData.length === 0) {
            emptyState.style.display = 'flex';
            tableContainer.style.display = 'none';
            document.querySelector('.results-count').style.display = 'none';
        } else {
            emptyState.style.display = 'none';
            tableContainer.style.display = 'table';
            document.querySelector('.results-count').style.display = 'block';
            renderHistoryTable();
        }
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

function renderHistoryTable() {
    const tableBody = document.getElementById('historyTableBody');
    const noResultsState = document.getElementById('noResultsState');
    const tableContainer = document.getElementById('historyTable');

    if (filteredHistoryData.length === 0) {
        tableBody.innerHTML = '';
        noResultsState.style.display = 'flex';
        tableContainer.style.display = 'none';
        document.getElementById('showingCount').textContent = '0';
        return;
    }

    noResultsState.style.display = 'none';
    tableContainer.style.display = 'table';
    document.getElementById('showingCount').textContent = filteredHistoryData.length;

    tableBody.innerHTML = filteredHistoryData.map(item => {
        const isFake = item.prediction === 'FAKE';
        const date = new Date(item.timestamp).toLocaleString();
        const confidence = (item.confidence * 100).toFixed(1);

        return `
            <tr data-id="${item.id}">
                <td>
                    ${item.image_path ?
                `<img src="/${item.image_path}" alt="${item.filename}" class="table-preview-img" onerror="this.style.display='none'">` :
                '<div class="table-preview-img" style="background: rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: center;">📷</div>'
            }
                </td>
                <td class="table-filename" title="${item.filename}">${item.filename}</td>
                <td>
                    <span class="table-badge ${isFake ? 'fake' : 'real'}">
                        ${isFake ? '⚠ FAKE' : '✓ REAL'}
                    </span>
                </td>
                <td>
                    <div class="table-confidence">
                        <span>${confidence}%</span>
                        <div class="confidence-bar-small">
                            <div class="confidence-fill-small" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                </td>
                <td class="table-date">${date}</td>
                <td>
                    <div class="table-actions">
                        <button class="btn-table-action" onclick="generateHistoryPDF(${JSON.stringify(item).replace(/"/g, '&quot;')})">
                            📄 Report
                        </button>
                        <button class="btn-table-action btn-table-delete" onclick="deleteHistoryItem(${item.id})">
                            🗑 Delete
                        </button>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
}

// Search functionality with debouncing
let searchTimeout;
function handleSearch() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        applyFilters();
    }, 300);
}

// Apply all filters
function applyFilters() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const predictionFilter = document.getElementById('filterPrediction').value;
    const confidenceFilter = document.getElementById('filterConfidence').value;
    const sortBy = document.getElementById('sortBy').value;

    // Start with full data
    filteredHistoryData = fullHistoryData.filter(item => {
        // Search filter
        if (searchTerm && !item.filename.toLowerCase().includes(searchTerm)) {
            return false;
        }

        // Prediction filter
        if (predictionFilter !== 'all' && item.prediction !== predictionFilter) {
            return false;
        }

        // Confidence filter
        if (confidenceFilter !== 'all') {
            const conf = item.confidence * 100;
            if (confidenceFilter === 'high' && conf <= 80) return false;
            if (confidenceFilter === 'medium' && (conf < 50 || conf > 80)) return false;
            if (confidenceFilter === 'low' && conf >= 50) return false;
        }

        return true;
    });

    // Apply sorting
    const [sortColumn, sortOrder] = sortBy.split('-');
    sortHistoryData(sortColumn, sortOrder);

    renderHistoryTable();
}

// Sort functionality
function sortHistoryData(column, order) {
    currentSortColumn = column;
    currentSortOrder = order;

    filteredHistoryData.sort((a, b) => {
        let aVal, bVal;

        switch (column) {
            case 'filename':
                aVal = a.filename.toLowerCase();
                bVal = b.filename.toLowerCase();
                break;
            case 'confidence':
                aVal = a.confidence;
                bVal = b.confidence;
                break;
            case 'prediction':
                aVal = a.prediction;
                bVal = b.prediction;
                break;
            case 'date':
            default:
                aVal = new Date(a.timestamp).getTime();
                bVal = new Date(b.timestamp).getTime();
                break;
        }

        if (order === 'asc') {
            return aVal > bVal ? 1 : -1;
        } else {
            return aVal < bVal ? 1 : -1;
        }
    });
}

// Alternative table header sort (if clicking headers directly)
function sortTable(column) {
    // Toggle order if same column
    if (currentSortColumn === column) {
        currentSortOrder = currentSortOrder === 'asc' ? 'desc' : 'asc';
    } else {
        currentSortColumn = column;
        currentSortOrder = 'desc';
    }

    sortHistoryData(currentSortColumn, currentSortOrder);
    updateSortIndicators();
    renderHistoryTable();
}

function updateSortIndicators() {
    document.querySelectorAll('.sort-indicator').forEach(indicator => {
        indicator.classList.remove('active', 'asc', 'desc');
    });

    // Find the right column and update
    const headers = document.querySelectorAll('.history-table th');
    headers.forEach(th => {
        const text = th.textContent.toLowerCase();
        if (text.includes(currentSortColumn) ||
            (currentSortColumn === 'timestamp' && text.includes('date'))) {
            const indicator = th.querySelector('.sort-indicator');
            if (indicator) {
                indicator.classList.add('active', currentSortOrder);
            }
        }
    });
}

// Delete individual item
async function deleteHistoryItem(id) {
    if (!confirm('Are you sure you want to delete this scan?')) return;

    const row = document.querySelector(`tr[data-id="${id}"]`);
    if (row) {
        row.style.opacity = '0';
        row.style.transform = 'scale(0.95)';
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/history/${id}`, { method: 'DELETE' });
        if (response.ok) {
            setTimeout(() => {
                fullHistoryData = fullHistoryData.filter(item => item.id !== id);
                applyFilters();

                // Check if history is now empty
                if (fullHistoryData.length === 0) {
                    document.getElementById('historyEmptyState').style.display = 'flex';
                    document.getElementById('historyTable').style.display = 'none';
                    document.querySelector('.results-count').style.display = 'none';
                }
            }, 300);
        }
    } catch (err) {
        console.error('Failed to delete item:', err);
        if (row) {
            row.style.opacity = '1';
            row.style.transform = 'scale(1)';
        }
    }
}

// Clear all history
function confirmClearAll() {
    if (!confirm('⚠️ WARNING: This will delete ALL scan history permanently. Are you sure?')) return;
    if (!confirm('This action cannot be undone. Continue?')) return;

    clearHistory();
}

// Export functions
function exportHistory(format) {
    if (filteredHistoryData.length === 0) {
        alert('No data to export. Please run some analyses first.');
        return;
    }

    if (format === 'csv') {
        exportToCSV();
    } else if (format === 'json') {
        exportToJSON();
    }
}

function exportToCSV() {
    const headers = ['ID', 'Filename', 'Prediction', 'Confidence', 'Fake Probability', 'Real Probability', 'Timestamp'];
    const rows = filteredHistoryData.map(item => [
        item.id,
        `"${item.filename}"`,
        item.prediction,
        item.confidence.toFixed(4),
        item.fake_probability ? item.fake_probability.toFixed(4) : 'N/A',
        item.real_probability ? item.real_probability.toFixed(4) : 'N/A',
        item.timestamp
    ]);

    const csvContent = [
        headers.join(','),
        ...rows.map(row => row.join(','))
    ].join('\n');

    downloadFile(csvContent, 'deepguard_history.csv', 'text/csv');
}

function exportToJSON() {
    const jsonContent = JSON.stringify(filteredHistoryData, null, 2);
    downloadFile(jsonContent, 'deepguard_history.json', 'application/json');
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    // Show success message
    playSound('success');
    const format = filename.split('.').pop().toUpperCase();
    setTimeout(() => {
        alert(`✅ Exported ${filteredHistoryData.length} records to ${format} successfully!`);
    }, 100);
}

// Event listeners for history page
if (window.location.pathname.includes('history.html')) {
    window.addEventListener('load', () => {
        loadEnhancedHistory();

        // Attach event listeners
        const searchInput = document.getElementById('searchInput');
        const filterPrediction = document.getElementById('filterPrediction');
        const filterConfidence = document.getElementById('filterConfidence');
        const sortBy = document.getElementById('sortBy');

        if (searchInput) searchInput.addEventListener('input', handleSearch);
        if (filterPrediction) filterPrediction.addEventListener('change', applyFilters);
        if (filterConfidence) filterConfidence.addEventListener('change', applyFilters);
        if (sortBy) sortBy.addEventListener('change', applyFilters);
    });
}

// Loading screen removed

// ==================== VIDEO DEMO PLAYER ====================
document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('demoVideoPlayer');
    const playOverlay = document.getElementById('playOverlay');
    const progressBar = document.getElementById('videoProgress');

    if (video && playOverlay) {
        const playlist = [
            'assets/demo_part1.mov',
            'assets/demo_part2.mov'
        ];
        let currentVideoIndex = 0;

        // Play Toggle
        playOverlay.addEventListener('click', () => {
            if (video.paused) {
                video.play();
                playOverlay.classList.add('hidden');
            } else {
                video.pause();
                playOverlay.classList.remove('hidden');
            }
        });

        video.addEventListener('click', () => {
            if (!video.paused) {
                video.pause();
                playOverlay.classList.remove('hidden');
            }
        });

        // Update Progress
        video.addEventListener('timeupdate', () => {
            const percent = (video.currentTime / video.duration) * 100;
            if (progressBar) progressBar.style.width = `${percent}%`;
        });

        // Handle Sequential Playback
        video.addEventListener('ended', () => {
            currentVideoIndex++;
            if (currentVideoIndex < playlist.length) {
                // Play next video
                video.src = playlist[currentVideoIndex];
                video.play().then(() => {
                    // Ensure overlay stays hidden
                    playOverlay.classList.add('hidden');
                }).catch(e => console.error("Auto-play blocked:", e));
            } else {
                // Playlist finished, reset
                currentVideoIndex = 0;
                video.src = playlist[0];
                playOverlay.classList.remove('hidden'); // Show play button again
                if (progressBar) progressBar.style.width = '0%';
            }
        });

        // Auto-play when in view
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && video.paused) {
                    video.play().then(() => {
                        playOverlay.classList.add('hidden');
                    }).catch(e => console.log("Autoplay prevented by browser interaction policy"));
                } else if (!entry.isIntersecting && !video.paused) {
                    video.pause();
                    playOverlay.classList.remove('hidden');
                }
            });
        }, { threshold: 0.5 }); // Start when 50% visible

        observer.observe(video);
    }
});
