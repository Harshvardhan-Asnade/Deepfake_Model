/**
 * Hero Reveal Effect
 * Adapts the fluid reveal effect for the hero section of the main landing page.
 */

class HeroFluidReveal {
    constructor() {
        this.container = document.getElementById('heroRevealContainer');
        this.canvas = document.getElementById('revealCanvas');

        if (!this.container || !this.canvas) {
            console.warn('HeroFluidReveal: Container or Canvas not found.');
            return;
        }

        this.ctx = this.canvas.getContext('2d');

        // Use the new IDs we added to index.html
        this.bgImg = document.getElementById('hero-img-bg');
        this.revealImg = document.getElementById('hero-img-reveal');

        // Initialize state
        this.width = this.container.offsetWidth;
        this.height = this.container.offsetHeight;

        // Mouse state (relative to container)
        this.mouse = { x: this.width / 2, y: this.height / 2 };
        this.targetMouse = { x: this.width / 2, y: this.height / 2 };
        this.isMouseOver = false;

        // Blob state
        this.blob = {
            x: this.width / 2,
            y: this.height / 2,
            vx: 0,
            vy: 0,
            radius: 250 // Slightly smaller for hero section if needed, or keep 300
        };

        // Configuration
        this.numPoints = 20;
        this.points = [];
        this.init();
    }

    init() {
        // Point class definition (same as before)
        this.Point = class {
            constructor(angle, radius) {
                this.angle = angle;
                this.baseRadius = radius;
                this.radius = radius;
                this.x = 0;
                this.y = 0;
                this.noiseOffset = Math.random() * 1000;
                this.speed = 0.002 + Math.random() * 0.003;
            }

            update(centerX, centerY, velocityX, velocityY, time) {
                const noise = Math.sin(time * this.speed + this.noiseOffset) * 20;
                const dirX = Math.cos(this.angle);
                const dirY = Math.sin(this.angle);
                const dot = dirX * velocityX + dirY * velocityY;
                const stretch = dot * 1.5;
                const currentRadius = this.baseRadius + noise - stretch;
                this.x = centerX + Math.cos(this.angle) * currentRadius;
                this.y = centerY + Math.sin(this.angle) * currentRadius;
            }
        };

        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Listen to window mouse events to avoid z-index blocking by hero content
        window.addEventListener('mousemove', (e) => this.onMouseMove(e));

        // Optional: We can still use container bounds to "pause" or hide if needed, 
        // but for a background effect, continuous tracking is usually better.
        // Removed container-specific enter/leave to prevent stuttering at edges of children.

        // Initialize points
        for (let i = 0; i < this.numPoints; i++) {
            const angle = (i / this.numPoints) * Math.PI * 2;
            this.points.push(new this.Point(angle, this.blob.radius));
        }

        // Start loop
        requestAnimationFrame((t) => this.render(t));
    }

    resize() {
        this.width = this.container.offsetWidth;
        this.height = this.container.offsetHeight;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
    }

    onMouseMove(e) {
        // Calculate mouse position relative to container
        const rect = this.container.getBoundingClientRect();
        this.targetMouse.x = e.clientX - rect.left;
        this.targetMouse.y = e.clientY - rect.top;
    }

    updateBlob() {
        const dx = this.targetMouse.x - this.blob.x;
        const dy = this.targetMouse.y - this.blob.y;

        // Ease
        const ease = 0.25;
        this.blob.vx += dx * ease;
        this.blob.vy += dy * ease;

        // Friction
        this.blob.vx *= 0.75;
        this.blob.vy *= 0.75;

        this.blob.x += this.blob.vx;
        this.blob.y += this.blob.vy;

        const velX = (this.targetMouse.x - this.blob.x) * 0.1;
        const velY = (this.targetMouse.y - this.blob.y) * 0.1;

        return { velX, velY };
    }

    drawBlobPath(time, velX, velY) {
        this.ctx.beginPath();
        this.points.forEach(p => p.update(this.blob.x, this.blob.y, velX, velY, time));

        const p0 = this.points[0];
        const pLast = this.points[this.points.length - 1];
        const midX = (p0.x + pLast.x) / 2;
        const midY = (p0.y + pLast.y) / 2;

        this.ctx.moveTo(midX, midY);

        for (let i = 0; i < this.points.length; i++) {
            const p = this.points[i];
            const nextP = this.points[(i + 1) % this.points.length];
            const nextMidX = (p.x + nextP.x) / 2;
            const nextMidY = (p.y + nextP.y) / 2;
            this.ctx.quadraticCurveTo(p.x, p.y, nextMidX, nextMidY);
        }
        this.ctx.closePath();
    }

    drawImageCover(img) {
        const imgRatio = img.width / img.height;
        const canvasRatio = this.width / this.height;
        let drawW, drawH, curX, curY;

        if (imgRatio > canvasRatio) {
            drawH = this.height;
            drawW = drawH * imgRatio;
            curX = (this.width - drawW) / 2;
            curY = 0;
        } else {
            drawW = this.width;
            drawH = drawW / imgRatio;
            curX = 0;
            curY = (this.height - drawH) / 2;
        }

        this.ctx.drawImage(img, curX, curY, drawW, drawH);
    }

    render(time) {
        this.ctx.clearRect(0, 0, this.width, this.height);

        // IMPORTANT: We do NOT draw the background image on the canvas.
        // The background image is an <img> tag in HTML (id="hero-img-bg").
        // The canvas sits ON TOP of it.
        // The canvas draws the "Reveal" image ONLY inside the blob.

        // 1. Update Physics
        const velocity = this.updateBlob();

        // 2. Create Mask and Draw Reveal
        this.ctx.save();
        this.drawBlobPath(time, velocity.velX, velocity.velY);
        this.ctx.clip();

        // Apply theme-based filter
        const theme = document.documentElement.getAttribute('data-theme');
        if (theme === 'light') {
            // Rotates Yellow (~60deg) to Blue (~240deg) -> +180deg
            this.ctx.filter = 'hue-rotate(190deg) brightness(1.1) saturate(1.2)';
        } else {
            this.ctx.filter = 'none';
        }

        if (this.revealImg && this.revealImg.complete) {
            this.drawImageCover(this.revealImg);
        }

        // Optional: Add a subtle border/glow to the reveal edge
        // this.ctx.lineWidth = 2;
        // this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        // this.ctx.stroke();

        this.ctx.restore();

        requestAnimationFrame((t) => this.render(t));
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait slightly to ensure images start loading? 
    // Actually window.onload is safer for images but DOMContentLoaded is faster for UI.
    // The class checks .complete so it handles loading.
    new HeroFluidReveal();
});
