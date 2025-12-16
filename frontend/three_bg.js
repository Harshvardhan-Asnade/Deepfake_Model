// 3D Background with Three.js
// Theme: Dark space with "Nano Yellow" stars/particles

function initThreeBackground() {
    const container = document.getElementById('canvas-container');
    if (!container) return;

    // SCENE
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.002);

    // CAMERA
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 1000);
    camera.position.z = 500;

    // RENDERER
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0); // Transparent background
    container.appendChild(renderer.domElement);

    // PARTICLES (STARS)
    const geometry = new THREE.BufferGeometry();
    const count = 2000;
    const vertices = [];
    const colors = [];

    const color1 = new THREE.Color(0xE3F514); // Nano Yellow
    const color2 = new THREE.Color(0xFFFFFF); // White

    for (let i = 0; i < count; i++) {
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
            color: 0xE3F514,
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
    const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, 0xE3F514, 0x333333);
    gridHelper.position.y = -200; // Floor level
    gridHelper.material.transparent = true;
    gridHelper.material.opacity = 0.15;
    scene.add(gridHelper);

    // MOUSE INTERACTION
    let mouseX = 0;
    let mouseY = 0;
    let targetX = 0;
    let targetY = 0;

    const windowHalfX = window.innerWidth / 2;
    const windowHalfY = window.innerHeight / 2;

    document.addEventListener('mousemove', (event) => {
        mouseX = (event.clientX - windowHalfX);
        mouseY = (event.clientY - windowHalfY);
    });

    // RESIZE HANDLER
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // SCROLL INTERACTION
    let scrollY = 0;
    let targetScrollY = 0;

    document.addEventListener('scroll', () => {
        scrollY = window.scrollY;
    });

    // ANIMATION LOOP
    function animate() {
        requestAnimationFrame(animate);

        // Smooth Scroll Interpolation
        targetScrollY += (scrollY - targetScrollY) * 0.05;

        // Mouse Parallax Calculation
        targetX = mouseX * 0.001;
        targetY = mouseY * 0.001;

        // 1. Particle System Rotation (Base + Mouse + Scroll)
        // Scroll adds a dramatic rotation around X axis (tumbling forward)
        particles.rotation.y += 0.0005;
        particles.rotation.x = targetScrollY * 0.0002;

        // Mouse interaction for rotation
        particles.rotation.y += 0.05 * (targetX - particles.rotation.y);
        particles.rotation.x += 0.05 * (targetY - particles.rotation.x);

        // 2. Camera Scroll Movement (Storytelling Effect)
        // As user scrolls down, camera flies 'into' the scene (Z-axis) and pans down (Y-axis)
        // Max Scroll assumptions: approx 3000px height
        const zoomFactor = targetScrollY * 0.1;
        camera.position.z = 500 - zoomFactor; // Move closer
        camera.position.y = -targetScrollY * 0.05; // Pan down subtly

        // Stop zooming too close
        if (camera.position.z < 100) camera.position.z = 100;

        // 3. Floating Shapes Animation (Enhanced with Scroll)
        shapes.forEach((shape, i) => {
            // Basic rotation
            shape.rotation.x += 0.002 * (i + 1);
            shape.rotation.y += 0.002 * (i + 1);

            // Scroll reaction: Shapes spread out or rotate faster
            // Adding a scroll-based rotation boost
            shape.rotation.z = targetScrollY * 0.001 * (i % 2 === 0 ? 1 : -1);
        });

        // 4. Grid Animation (Infinite Scroll)
        // Move grid towards camera (z-axis)
        gridHelper.position.z = (Date.now() * 0.05) % (gridSize / gridDivisions);
        // Also react to scroll speed (optional warp effect)
        gridHelper.position.z += targetScrollY * 0.5;
        // Keep it looping within one cell size to look infinite
        const cell = gridSize / gridDivisions;
        if (gridHelper.position.z > cell) gridHelper.position.z -= cell;

        renderer.render(scene, camera);
    }

    animate();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initThreeBackground);
