// 3D Background with Three.js
// Theme: Dark space with "Nano Yellow" stars/particles

function initThreeBackground() {
    const container = document.getElementById('canvas-container');
    if (!container) return;

    // PERFORMANCE OPTIMIZATION: Reduce count on mobile
    const isMobile = window.innerWidth < 768;
    const particleCount = isMobile ? 400 : 1200; // significantly fewer particles on mobile

    // SCENE
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.002);

    // CAMERA
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 1000);
    camera.position.z = 500;

    // RENDERER - PERFORMANCE TUNING
    const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: !isMobile, // Disable antialias on mobile for performance
        powerPreference: "high-performance" // Hint to browser
    });

    // Cap pixel ratio to 2 to avoid 9x rendering on 3x screens
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0); // Transparent background
    container.appendChild(renderer.domElement);

    // Theme Check
    const getThemeColors = () => {
        const theme = localStorage.getItem('theme') || 'dark';
        if (theme === 'light') {
            return {
                primary: 0x00AEEF, // Cyan
                secondary: 0x0044CC, // Blue
                gridPrimary: 0x00AEEF, // Cyan Grid
                gridSecondary: 0xE0E0E0 // Light Grey Grid
            };
        }
        return {
            primary: 0xE3F514, // Nano Yellow
            secondary: 0xFFFFFF, // White
            gridPrimary: 0xE3F514, // Nano Yellow Grid
            gridSecondary: 0x333333 // Dark Grey Grid
        };
    };

    let themeColors = getThemeColors();

    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];

    const color1 = new THREE.Color(themeColors.primary);
    const color2 = new THREE.Color(themeColors.secondary);

    for (let i = 0; i < particleCount; i++) {
        // Random position
        const x = (Math.random() - 0.5) * 2000;
        const y = (Math.random() - 0.5) * 2000;
        const z = (Math.random() - 0.5) * 2000;
        vertices.push(x, y, z);

        // Random color mix
        const mixedColor = color1.clone().lerp(color2, Math.random() * 0.5);
        colors.push(mixedColor.r, mixedColor.g, mixedColor.b);
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 2,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        sizeAttenuation: true
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // GEOMETRIC SHAPES (Floating low-poly meshes)
    const shapeGroup = new THREE.Group();
    scene.add(shapeGroup);

    function addFloatingShape(type, x, y, z, size) {
        let geometry;
        if (type === 'icosahedron') geometry = new THREE.IcosahedronGeometry(size, 0);
        else if (type === 'octahedron') geometry = new THREE.OctahedronGeometry(size, 0);

        const material = new THREE.MeshBasicMaterial({
            color: themeColors.primary,
            wireframe: true,
            transparent: true,
            opacity: 0.15
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y, z);
        shapeGroup.add(mesh);
        return mesh;
    }

    // Add a few floating shapes
    const shapes = [];
    shapes.push(addFloatingShape('icosahedron', -300, 100, -200, 60));
    shapes.push(addFloatingShape('octahedron', 400, -150, -300, 80));
    shapes.push(addFloatingShape('icosahedron', 0, 200, -400, 40));

    // 3. INTERACTIVE 3D GRID FLOOR
    const gridSize = 2000;
    const gridDivisions = 40;
    // Change const to let to allow reassignment
    let gridHelper = new THREE.GridHelper(gridSize, gridDivisions, themeColors.gridPrimary, themeColors.gridSecondary);
    gridHelper.position.y = -200; // Floor level
    gridHelper.material.transparent = true;
    gridHelper.material.opacity = 0.15;
    scene.add(gridHelper);

    // Watch for theme changes
    // Store initial theme to avoid redundant updates on load
    let currentTheme = localStorage.getItem('theme') || 'dark';

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
                const newTheme = document.documentElement.getAttribute('data-theme');

                // Prevent infinite loop relative to initial set or same-value updates
                if (newTheme === currentTheme) return;
                currentTheme = newTheme;

                const isLight = newTheme === 'light';

                const newPrim = new THREE.Color(isLight ? 0x00AEEF : 0xE3F514);
                const newSec = new THREE.Color(isLight ? 0x0044CC : 0xFFFFFF);

                // Update Particles
                const newColors = [];
                for (let i = 0; i < particleCount; i++) {
                    const mixedColor = newPrim.clone().lerp(newSec, Math.random() * 0.5);
                    newColors.push(mixedColor.r, mixedColor.g, mixedColor.b);
                }
                particles.geometry.setAttribute('color', new THREE.Float32BufferAttribute(newColors, 3));
                particles.geometry.attributes.color.needsUpdate = true;

                // Update Shapes
                shapes.forEach(shape => {
                    shape.material.color.set(newPrim);
                });

                // Update Grid
                scene.remove(gridHelper);
                // Create new grid using standard ThreeJS helper for fixed geometry colors
                gridHelper = new THREE.GridHelper(gridSize, gridDivisions, isLight ? 0x00AEEF : 0xE3F514, isLight ? 0xE0E0E0 : 0x333333);
                gridHelper.position.y = -200;
                gridHelper.material.transparent = true;
                gridHelper.material.opacity = 0.15;
                scene.add(gridHelper);
            }
        });
    });

    observer.observe(document.documentElement, { attributes: true });


    // MOUSE INTERACTION
    let mouseX = 0;
    let mouseY = 0;
    let targetX = 0;
    let targetY = 0;

    const windowHalfX = window.innerWidth / 2;
    const windowHalfY = window.innerHeight / 2;

    document.addEventListener('mousemove', (event) => {
        // Optimize: use requestAnimationFrame for mouse updates if needed, but direct is usually fine for coordinates
        mouseX = (event.clientX - windowHalfX);
        mouseY = (event.clientY - windowHalfY);
    });

    // RESIZE HANDLER (THROTTLED)
    let resizeTimeout;
    window.addEventListener('resize', () => {
        if (!resizeTimeout) {
            resizeTimeout = setTimeout(() => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);

                // Update constraints for mouse calc
                // windowHalfX = window.innerWidth / 2; // const variable can't be reassigned, generally okay not to update center exactly on resize for this effect

                resizeTimeout = null;
            }, 100);
        }
    });

    // SCROLL INTERACTION
    let scrollY = 0;
    let targetScrollY = 0;

    // Use passive listener for better scroll performance
    document.addEventListener('scroll', () => {
        scrollY = window.scrollY;
    }, { passive: true });

    // ANIMATION LOOP
    function animate() {
        requestAnimationFrame(animate);

        // Smooth Scroll Interpolation
        targetScrollY += (scrollY - targetScrollY) * 0.05;

        // Mouse Parallax Calculation
        targetX = mouseX * 0.001;
        targetY = mouseY * 0.001;

        // 1. Particle System Rotation
        particles.rotation.y += 0.0005;
        particles.rotation.x = targetScrollY * 0.0002;

        // Mouse interaction for rotation
        particles.rotation.y += 0.05 * (targetX - particles.rotation.y);
        particles.rotation.x += 0.05 * (targetY - particles.rotation.x);

        // 2. Camera Scroll Movement
        const zoomFactor = targetScrollY * 0.1;
        camera.position.z = 500 - zoomFactor; // Move closer
        camera.position.y = -targetScrollY * 0.05; // Pan down subtly

        // Stop zooming too close
        if (camera.position.z < 100) camera.position.z = 100;

        // 3. Floating Shapes Animation
        for (let i = 0; i < shapes.length; i++) {
            const shape = shapes[i];
            shape.rotation.x += 0.002 * (i + 1);
            shape.rotation.y += 0.002 * (i + 1);
            shape.rotation.z = targetScrollY * 0.001 * (i % 2 === 0 ? 1 : -1);
        }

        // 4. Grid Animation (Infinite Scroll)
        // Move grid towards camera (z-axis)
        // Using modulo based on local vars to avoid Date.now() overhead if high precision distinctness isn't needed
        // But Date.now() is fine.
        gridHelper.position.z = (Date.now() * 0.05) % (gridSize / gridDivisions);
        gridHelper.position.z += targetScrollY * 0.5;

        const cell = gridSize / gridDivisions;
        if (gridHelper.position.z > cell) gridHelper.position.z -= cell;

        renderer.render(scene, camera);
    }

    animate();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initThreeBackground);
