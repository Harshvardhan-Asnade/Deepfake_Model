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
        particlesJS('particles-js', {
            particles: {
                number: { value: 60, density: { enable: true, value_area: 800 } },
                color: { value: '#E3F514' },
                shape: { type: 'circle' },
                opacity: { value: 0.3, random: true },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#E3F514',
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
const magneticBtns = document.querySelectorAll('.btn-primary, .btn-hero-primary, .btn-hero-secondary, .nav-link');

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
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    playSound('scan'); // Trigger scan sound

    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
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
            <h3 style="margin-top: -20px;">Analyzing Media...</h3>
            <p>Running DeepGuard detection pipeline</p>
        `;
        resultsSection.appendChild(loader);
    }
    loader.style.display = 'block';

    try {
        // Call Backend
        const formData = new FormData();
        formData.append('file', file);

        const startTime = performance.now(); // Start timer

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Analysis failed');


        const result = await response.json();

        const endTime = performance.now(); // End timer
        const duration = ((endTime - startTime) / 1000).toFixed(2);
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
                    'rgba(255, 255, 255, 0.1)'  // Real (White transparent)
                ],
                borderColor: [
                    'rgba(227, 245, 20, 1)',
                    'rgba(255, 255, 255, 0.2)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#fff' }
                }
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
}

function resetAnalysis() {
    document.getElementById('fileInput').value = '';
    document.getElementById('previewArea').style.display = 'none';
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
                    animateValue(statValue, 0, 99.8, 2000);
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
async function generatePDFReport() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    // Capture the Dashboard
    const element = document.querySelector('.split-container');
    const originalBg = element.style.background;
    element.style.background = '#000000'; // Ensure dark background for capture

    try {
        const canvas = await html2canvas(element, {
            scale: 2,
            useCORS: true,
            logging: false,
            backgroundColor: '#000000'
        });

        element.style.background = originalBg; // Restore

        const imgData = canvas.toDataURL('image/png');
        const imgWidth = 190; // A4 width in mm minus margins
        const pageHeight = 295;
        const imgHeight = (canvas.height * imgWidth) / canvas.width;

        // Add Title Header
        doc.setFillColor(227, 245, 20); // Electric Yellow
        doc.rect(0, 0, 210, 30, 'F');
        doc.setTextColor(0, 0, 0);
        doc.setFontSize(22);
        doc.setFont('helvetica', 'bold');
        doc.text("DeepGuard Forensic Report", 105, 20, { align: "center" });

        doc.setFontSize(12);
        doc.text(`Generated: ${new Date().toLocaleString()}`, 105, 28, { align: "center" });

        // Add Analysis Image
        doc.addImage(imgData, 'PNG', 10, 40, imgWidth, imgHeight);

        // Add Footer
        doc.setFontSize(10);
        doc.setTextColor(100, 100, 100);

    } catch (err) {
        console.error("PDF Generation Failed:", err);
        alert("Could not generate report. Please try again.");
    }
}

// ==================== HISTORY LOGIC ====================
async function loadHistory() {
    const historyList = document.getElementById('historyList');
    const emptyState = document.getElementById('historyEmptyState');

    if (!historyList) return;

    try {
        const response = await fetch('/api/history');
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
        const response = await fetch(`/api/history/${scanId}`, { method: 'DELETE' });
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
        await fetch('/api/history', { method: 'DELETE' });
        loadHistory(); // Reload UI
    } catch (err) {
        console.error('Failed to clear history:', err);
    }
}

// Auto-load history on history.html
if (window.location.pathname.includes('history.html')) {
    window.addEventListener('load', loadHistory);
}
