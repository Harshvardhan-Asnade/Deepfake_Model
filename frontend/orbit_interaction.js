const featureData = {
    fusion: {
        title: "Hybrid Fusion Architecture",
        desc: "Combines EfficientNetV2 for spatial details and Swin Transformer V2 for global context with frequency domain analysis.",
        tags: ["CNN-ViT", "Multi-Modal", "Spatial-Temporal"]
    },
    realtime: {
        title: "Real-Time Analysis",
        desc: "Lightning-fast detection processing thousands of images per minute with GPU acceleration and optimized pipelines.",
        tags: ["CUDA", "Batch Process", "Low Latency"]
    },
    accuracy: {
        title: "99% Detection Accuracy",
        desc: "Industry-leading precision in detecting AI-generated and manipulated media across various generation methods.",
        tags: ["Verified", "Tested", "SOTA"]
    },
    analytics: {
        title: "Advanced Analytics",
        desc: "Comprehensive reports with confidence scores, heatmaps, and detailed forensic analysis for every scan.",
        tags: ["Reports", "Forensics", "Heatmaps"]
    }
};

function initRobustOrbit() {
    const nodes = document.querySelectorAll('.orbit-node');
    const ring = document.querySelector('.orbit-ring');
    const card = document.getElementById('orbitInfoCard');
    const title = document.getElementById('orbitTitle');
    const desc = document.getElementById('orbitDesc');
    const tagsContainer = document.getElementById('orbitTags');
    const hub = document.querySelector('.orbit-hub-inner');

    if (!nodes.length || !ring || !card) return;

    let hoverTimeout;
    let isHoveringNode = false;
    let isHoveringCard = false;

    const updateState = () => {
        if (isHoveringNode || isHoveringCard) {
            clearTimeout(hoverTimeout);
            card.classList.add('active');
            ring.classList.add('paused');
            if (hub) {
                hub.style.boxShadow = "0 0 70px var(--accent-yellow)";
                hub.textContent = "🔍";
            }
        } else {
            hoverTimeout = setTimeout(() => {
                if (!isHoveringNode && !isHoveringCard) {
                    card.classList.remove('active');
                    ring.classList.remove('paused');
                    if (hub) {
                        hub.style.boxShadow = "";
                        hub.textContent = "🛡️";
                    }
                }
            }, 150);
        }
    };

    nodes.forEach(node => {
        node.addEventListener('mouseenter', () => {
            isHoveringNode = true;
            const id = node.getAttribute('data-id');
            const data = featureData[id];
            if (data) {
                title.textContent = data.title;
                desc.textContent = data.desc;
                tagsContainer.innerHTML = data.tags.map(t => `<span class="orbit-tag">${t}</span>`).join('');
            }
            updateState();
        });

        node.addEventListener('mouseleave', () => {
            isHoveringNode = false;
            updateState();
        });
    });

    // Keeping it frozen when mouse is over the card itself
    card.addEventListener('mouseenter', () => {
        isHoveringCard = true;
        updateState();
    });

    card.addEventListener('mouseleave', () => {
        isHoveringCard = false;
        updateState();
    });

    // Stability: Force animation refresh
    requestAnimationFrame(() => {
        ring.style.animation = 'none';
        void ring.offsetWidth;
        ring.style.animation = 'orbit-main-v4 45s linear infinite';

        document.querySelectorAll('.node-content').forEach(nc => {
            nc.style.animation = 'none';
            void nc.offsetWidth;
            nc.style.animation = 'counter-v4 45s linear infinite';
        });
    });
}

// Multi-method startup
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initRobustOrbit);
} else {
    initRobustOrbit();
}
window.addEventListener('load', initRobustOrbit);
