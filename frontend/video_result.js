// Video Result Page - Enhanced Functionality
// ==========================================

let videoPlayer;
let currentResult = null;
let chart = null;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize
    videoPlayer = document.getElementById('videoPlayer');

    // Retrieve results from localStorage
    const resultData = localStorage.getItem('video_analysis_result');

    if (!resultData) {
        alert('No analysis results found. Redirecting to home.');
        window.location.href = 'index.html';
        return;
    }

    currentResult = JSON.parse(resultData);

    // Initialize everything
    initializeVideoPlayer();
    populateUI(currentResult);
    setupVideoControls();
    setupDownloadButton();

    // Hide loading overlay
    setTimeout(() => {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }, 500);
});

// ==========================================
// VIDEO PLAYER INITIALIZATION
// ==========================================

function initializeVideoPlayer() {
    // Set video source if available
    // Try multiple sources in order of priority
    if (currentResult && currentResult.video_url) {
        // Ensure path starts with / if relative
        let src = currentResult.video_url;
        if (!src.startsWith('http') && !src.startsWith('/')) {
            src = '/' + src;
        }
        videoPlayer.src = src;
    } else if (currentResult && currentResult.image_path) {
        // Fallback or Image Path logic
        let src = currentResult.image_path;
        if (!src.startsWith('http') && !src.startsWith('/')) {
            src = '/' + src;
        }
        videoPlayer.src = src;
    } else {
        console.warn('No video source available in result data');
        // Show a placeholder message if no video is available
        const videoWrapper = document.querySelector('.video-wrapper');
        if (videoWrapper) {
            const placeholder = document.createElement('div');
            placeholder.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                color: rgba(255, 255, 255, 0.6);
                z-index: 5;
            `;
            placeholder.innerHTML = `
                <i class="fas fa-video-slash" style="font-size: 4rem; margin-bottom: 1rem; display: block; color: rgba(227, 245, 20, 0.3);"></i>
                <p style="font-size: 1.1rem;">Video source not available</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">The video file could not be loaded</p>
            `;
            videoWrapper.appendChild(placeholder);
        }
    }

    // Video event listeners
    videoPlayer.addEventListener('loadedmetadata', () => {
        updateDuration();
    });

    videoPlayer.addEventListener('timeupdate', () => {
        updateProgress();
        updateTimeDisplay();
    });

    videoPlayer.addEventListener('ended', () => {
        document.querySelector('#playPauseBtn i').className = 'fas fa-redo';
    });

    // Error handling
    videoPlayer.addEventListener('error', (e) => {
        console.error('Video load error:', e);
        console.error('Failed source:', videoPlayer.src);
        const videoWrapper = document.querySelector('.video-wrapper');
        if (videoWrapper) {
            const errorMsg = document.createElement('div');
            errorMsg.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                color: rgba(255, 51, 51, 0.8);
                z-index: 5;
                background: rgba(0,0,0,0.7);
                padding: 20px;
                border-radius: 10px;
            `;
            errorMsg.innerHTML = `
                <i class="fas fa-exclamation-triangle" style="font-size: 4rem; margin-bottom: 1rem; display: block;"></i>
                <p style="font-size: 1.1rem;">Failed to load video</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">The video format may not be supported</p>
                <p style="font-size: 0.8rem; color: #aaa; margin-top: 10px; word-break: break-all;">Source: ${videoPlayer.src}</p>
            `;
            videoWrapper.appendChild(errorMsg);
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;

        switch (e.key) {
            case ' ':
                e.preventDefault();
                togglePlay();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 5);
                break;
            case 'ArrowRight':
                e.preventDefault();
                videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + 5);
                break;
            case 'f':
                toggleFullscreen();
                break;
            case 'm':
                toggleMute();
                break;
        }
    });
}

// ==========================================
// VIDEO CONTROLS
// ==========================================

function setupVideoControls() {
    // Play/Pause
    const playPauseBtn = document.getElementById('playPauseBtn');
    playPauseBtn.addEventListener('click', togglePlay);

    // Progress bar
    const progressContainer = document.getElementById('progressContainer');
    progressContainer.addEventListener('click', seek);

    // Frame navigation
    document.getElementById('prevFrameBtn').addEventListener('click', () => {
        videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - (1 / 30)); // Assuming 30fps
    });

    document.getElementById('nextFrameBtn').addEventListener('click', () => {
        videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + (1 / 30));
    });

    // Volume
    const volumeSlider = document.getElementById('volumeSlider');
    const muteBtn = document.getElementById('muteBtn');

    volumeSlider.addEventListener('input', (e) => {
        const volume = e.target.value / 100;
        videoPlayer.volume = volume;
        updateVolumeIcon(volume);
    });

    muteBtn.addEventListener('click', toggleMute);

    // Speed control
    const speedBtn = document.getElementById('speedBtn');
    const speedMenu = document.getElementById('speedMenu');

    speedBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        speedMenu.classList.toggle('active');
    });

    document.addEventListener('click', () => {
        speedMenu.classList.remove('active');
    });

    document.querySelectorAll('.speed-option').forEach(option => {
        option.addEventListener('click', (e) => {
            e.stopPropagation();
            const speed = parseFloat(e.target.dataset.speed);
            videoPlayer.playbackRate = speed;

            document.querySelectorAll('.speed-option').forEach(opt => opt.classList.remove('active'));
            e.target.classList.add('active');
            speedMenu.classList.remove('active');
        });
    });

    // Fullscreen
    document.getElementById('fullscreenBtn').addEventListener('click', toggleFullscreen);

    // Video click to play/pause
    videoPlayer.addEventListener('click', togglePlay);
}

function togglePlay() {
    if (videoPlayer.paused) {
        videoPlayer.play();
        document.querySelector('#playPauseBtn i').className = 'fas fa-pause';
    } else {
        videoPlayer.pause();
        document.querySelector('#playPauseBtn i').className = 'fas fa-play';
    }
}

function seek(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    videoPlayer.currentTime = percent * videoPlayer.duration;
}

function toggleMute() {
    videoPlayer.muted = !videoPlayer.muted;
    updateVolumeIcon(videoPlayer.muted ? 0 : videoPlayer.volume);
}

function updateVolumeIcon(volume) {
    const muteBtn = document.querySelector('#muteBtn i');
    if (volume === 0) {
        muteBtn.className = 'fas fa-volume-mute';
    } else if (volume < 0.5) {
        muteBtn.className = 'fas fa-volume-down';
    } else {
        muteBtn.className = 'fas fa-volume-up';
    }
}

function toggleFullscreen() {
    const container = document.querySelector('.video-player-container');

    if (!document.fullscreenElement) {
        container.requestFullscreen().catch(err => {
            console.error('Fullscreen error:', err);
        });
        document.querySelector('#fullscreenBtn i').className = 'fas fa-compress';
    } else {
        document.exitFullscreen();
        document.querySelector('#fullscreenBtn i').className = 'fas fa-expand';
    }
}

function updateProgress() {
    const percent = (videoPlayer.currentTime / videoPlayer.duration) * 100;
    document.getElementById('progressBar').style.width = percent + '%';
}

function updateTimeDisplay() {
    document.getElementById('currentTime').textContent = formatTime(videoPlayer.currentTime);
}

function updateDuration() {
    document.getElementById('duration').textContent = formatTime(videoPlayer.duration);
}

function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ==========================================
// UI POPULATION
// ==========================================

function populateUI(result) {
    // Verdict
    const isFake = result.prediction === 'FAKE';
    const title = document.getElementById('verdictTitle');
    const bar = document.getElementById('confidenceBar');
    const val = document.getElementById('confidenceValue');

    title.textContent = isFake ? 'FAKE VIDEO DETECTED' : 'AUTHENTIC VIDEO';
    title.style.color = isFake ? '#ff3333' : '#10B981';

    const conf = (result.confidence * 100).toFixed(1);

    // Animate confidence bar
    setTimeout(() => {
        bar.style.width = `${conf}%`;
    }, 100);

    bar.className = `meter-fill ${isFake ? 'fill-fake' : 'fill-real'}`;
    val.textContent = `${conf}% Confidence`;

    // Update verdict icon
    const verdictIcon = document.querySelector('.verdict-icon');
    if (isFake) {
        verdictIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        verdictIcon.style.background = 'linear-gradient(135deg, #ff3333 0%, #cc0000 100%)';
    } else {
        verdictIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
        verdictIcon.style.background = 'linear-gradient(135deg, #10B981 0%, #059669 100%)';
    }

    // Stats with animation
    animateValue('videoDuration', 0, result.duration, `${result.duration.toFixed(1)}s`, 1000);
    animateValue('framesProcessed', 0, result.processed_frames, result.processed_frames, 1000);
    animateValue('suspiciousCount', 0, result.suspicious_frames.length, result.suspicious_frames.length, 1000);
    animateValue('avgProb', 0, result.avg_fake_prob * 100, `${(result.avg_fake_prob * 100).toFixed(1)}%`, 1000);

    // Notes
    const notes = document.getElementById('analysisNotes');
    if (isFake) {
        notes.innerHTML = `
            <div style="display: flex; align-items: start; gap: 12px; padding: 1rem; background: rgba(255, 51, 51, 0.1); border-left: 3px solid #ff3333; border-radius: 8px; margin-bottom: 1rem;">
                <i class="fas fa-exclamation-circle" style="color: #ff3333; font-size: 24px; margin-top: 2px;"></i>
                <div>
                    <strong style="color: #ff3333; font-size: 1.1rem;">⚠️ Manipulation Detected</strong><br>
                    <span style="margin-top: 8px; display: block;">High probability of synthetic frames found. The model detected artifacts consistent with deepfake generation techniques in ${result.fake_frame_ratio ? (result.fake_frame_ratio * 100).toFixed(0) : 0}% of the sampled frames.</span>
                </div>
            </div>
            <p><strong>Detected Artifacts:</strong></p>
            <ul style="margin-left: 1.5rem; margin-top: 0.5rem; line-height: 1.8;">
                <li>Inconsistent temporal patterns across frames</li>
                <li>Frequency domain anomalies</li>
                <li>Unnatural facial features or lighting</li>
            </ul>
        `;
    } else {
        notes.innerHTML = `
            <div style="display: flex; align-items: start; gap: 12px; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10B981; border-radius: 8px; margin-bottom: 1rem;">
                <i class="fas fa-check-circle" style="color: #10B981; font-size: 24px; margin-top: 2px;"></i>
                <div>
                    <strong style="color: #10B981; font-size: 1.1rem;">✓ Authentic Media</strong><br>
                    <span style="margin-top: 8px; display: block;">No significant signs of manipulation were detected. Frame consistency is high across all analyzed segments.</span>
                </div>
            </div>
            <p><strong>Quality Indicators:</strong></p>
            <ul style="margin-left: 1.5rem; margin-top: 0.5rem; line-height: 1.8;">
                <li>Consistent temporal flow</li>
                <li>Natural frequency patterns</li>
                <li>Authentic visual characteristics</li>
            </ul>
        `;
    }

    // Chart
    if (result.timeline) {
        renderChart(result.timeline);
        addFrameMarkers(result.timeline, result.duration);
    }

    // Frame Grid
    if (result.timeline) {
        renderFrameGrid(result.timeline, result.duration);
    }
}

// ==========================================
// CHART RENDERING
// ==========================================

function renderChart(timeline) {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    const times = timeline.map(t => formatTime(t.time));
    const probs = timeline.map(t => t.prob);

    // Destroy existing chart if any
    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [{
                label: 'Fake Probability',
                data: probs,
                borderColor: '#E3F514',
                backgroundColor: 'rgba(227, 245, 20, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 8,
                pointBackgroundColor: '#E3F514',
                pointBorderColor: '#000',
                pointBorderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const index = elements[0].index;
                    const time = timeline[index].time;
                    videoPlayer.currentTime = time;
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: {
                        color: 'rgba(255,255,255,0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#888',
                        callback: function (value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Fake Probability',
                        color: '#888'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255,255,255,0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#888',
                        maxTicksLimit: 10
                    },
                    title: {
                        display: true,
                        text: 'Time',
                        color: '#888'
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    titleColor: '#E3F514',
                    bodyColor: '#fff',
                    borderColor: 'rgba(227, 245, 20, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        title: (context) => `Time: ${context[0].label}`,
                        label: (ctx) => `Probability: ${(ctx.raw * 100).toFixed(1)}%`,
                        afterLabel: (ctx) => {
                            const prob = ctx.raw;
                            if (prob > 0.5) {
                                return 'Status: Suspicious ⚠️';
                            } else {
                                return 'Status: Clean ✓';
                            }
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// ==========================================
// FRAME MARKERS
// ==========================================

function addFrameMarkers(timeline, duration) {
    const progressContainer = document.getElementById('progressContainer');

    timeline.forEach(frame => {
        if (frame.prob > 0.5) { // Suspicious frames
            const marker = document.createElement('div');
            marker.className = 'frame-marker';
            marker.style.left = ((frame.time / duration) * 100) + '%';
            marker.title = `Suspicious frame at ${formatTime(frame.time)}`;

            marker.addEventListener('click', (e) => {
                e.stopPropagation();
                videoPlayer.currentTime = frame.time;
            });

            progressContainer.appendChild(marker);
        }
    });
}

// ==========================================
// FRAME GRID
// ==========================================

function renderFrameGrid(timeline, duration) {
    const frameGrid = document.getElementById('frameGrid');
    frameGrid.innerHTML = '';

    timeline.forEach((frame, index) => {
        const frameItem = document.createElement('div');
        frameItem.className = 'frame-item' + (frame.prob > 0.5 ? ' suspicious' : '');

        const thumbContent = frame.thumbnail
            ? `<img src="data:image/jpeg;base64,${frame.thumbnail}" style="width: 100%; height: 100%; object-fit: cover;">`
            : `<i class="fas fa-film" style="font-size: 2rem;"></i>`;

        frameItem.innerHTML = `
            <div class="frame-thumbnail" style="display: flex; align-items: center; justify-content: center; color: #666; font-size: 0.9rem; overflow: hidden; background: #000;">
                ${thumbContent}
            </div>
            <div class="frame-badge ${frame.prob > 0.5 ? 'suspicious' : 'clean'}">
                ${frame.prob > 0.5 ? '⚠️ ' + (frame.prob * 100).toFixed(0) + '%' : '✓ ' + ((1 - frame.prob) * 100).toFixed(0) + '%'}
            </div>
            <div class="frame-info">
                <span class="frame-time">${formatTime(frame.time)}</span>
                <span class="frame-confidence">${(frame.prob * 100).toFixed(1)}%</span>
            </div>
        `;

        frameItem.addEventListener('click', () => {
            videoPlayer.currentTime = frame.time;
            videoPlayer.play();
        });

        frameGrid.appendChild(frameItem);
    });
}

// ==========================================
// ANIMATIONS
// ==========================================

function animateValue(id, start, end, suffix, duration) {
    const element = document.getElementById(id);
    const startTime = performance.now();
    const isPercentage = typeof suffix === 'string' && suffix.includes('%');

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * easeOut;

        if (isPercentage) {
            element.textContent = suffix;
        } else if (typeof suffix === 'string') {
            element.textContent = suffix;
        } else {
            element.textContent = Math.floor(current);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = suffix;
        }
    }

    requestAnimationFrame(update);
}

// ==========================================
// DOWNLOAD REPORT
// ==========================================

function setupDownloadButton() {
    const downloadBtn = document.getElementById('downloadReportBtn');

    downloadBtn.addEventListener('click', async () => {
        const originalText = downloadBtn.innerHTML;
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating Report...';
        downloadBtn.disabled = true;

        try {
            await generatePDFReport();
            downloadBtn.innerHTML = '<i class="fas fa-check"></i> Report Downloaded!';

            setTimeout(() => {
                downloadBtn.innerHTML = originalText;
                downloadBtn.disabled = false;
            }, 3000);
        } catch (error) {
            console.error('Error generating report:', error);
            downloadBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';

            alert(`Failed to generate report: ${error.message || error}\nPlease check console for details.`);

            setTimeout(() => {
                downloadBtn.innerHTML = originalText;
                downloadBtn.disabled = false;
            }, 3000);
        }
    });
}

async function generatePDFReport() {
    const reportContainer = document.getElementById('reportContainer');
    const isFake = currentResult.prediction === 'FAKE';
    const accentColor = isFake ? '#ff3333' : '#10B981';

    // 1. Construct Report HTML
    const dateStr = new Date().toLocaleDateString('en-US', {
        year: 'numeric', month: 'long', day: 'numeric',
        hour: '2-digit', minute: '2-digit'
    });

    // Get frames for report (max 8)
    // Prioritize suspicious frames if fake, otherwise spread out frames
    let reportFrames = [];
    if (isFake && currentResult.suspicious_frames && currentResult.suspicious_frames.length > 0) {
        // Take up to 8 suspicious frames
        reportFrames = currentResult.suspicious_frames
            .slice(0, 8)
            .map(idx => currentResult.timeline[idx])
            .filter(frame => frame !== undefined);
    } else if (currentResult.timeline && currentResult.timeline.length > 0) {
        // Take up to 8 evenly spaced frames
        const step = Math.max(1, Math.floor(currentResult.timeline.length / 8));
        for (let i = 0; i < currentResult.timeline.length && reportFrames.length < 8; i += step) {
            if (currentResult.timeline[i]) {
                reportFrames.push(currentResult.timeline[i]);
            }
        }
    }

    const framesHTML = reportFrames.map(frame => {
        if (!frame) return '';
        return `
        <div class="report-frame-item">
            ${frame.thumbnail ? `<img src="data:image/jpeg;base64,${frame.thumbnail}">` : ''}
            <div class="report-frame-badge" style="background: ${frame.prob > 0.5 ? '#ff3333' : '#10B981'}">
                ${(frame.prob * 100).toFixed(0)}%
            </div>
        </div>
    `}).join('');

    // Capture chart as image
    const chartCanvas = document.getElementById('timelineChart');
    const chartImg = chartCanvas.toDataURL('image/png');

    // Capture video preview (thumbnail of current frame)
    // We can use the first frame of the timeline if available, or just a placeholder if video element is cross-origin restricted
    // Ideally, we'd capture the video element, but that's often blocked by CORS or returns black.
    // Let's use the thumbnail of the most significant frame from the timeline as the 'video preview'
    // Find the first frame with a valid thumbnail in reportFrames, or fallback to any frame in timeline
    let mainPreviewThumb = '';

    // Helper to check for thumbnail
    const hasThumb = (f) => f && f.thumbnail;

    // 1. Try report frames (suspicious/key frames)
    const previewFrame = reportFrames.find(hasThumb) || currentResult.timeline.find(hasThumb);

    if (previewFrame) {
        mainPreviewThumb = `data:image/jpeg;base64,${previewFrame.thumbnail}`;
    }

    reportContainer.innerHTML = `
        <div class="report-header">
            <div class="report-logo">
                <img src="logo.png" alt="DeepGuard Logo">
                <span>DeepGuard Analysis Report</span>
            </div>
            <div class="report-meta">
                <p>Generated on: ${dateStr}</p>
                <p>Ref ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()}</p>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 2fr 3fr; gap: 30px; margin-bottom: 30px;">
            <!-- Video Preview Section -->
            <div class="report-card" style="margin: 0; padding: 10px; display: flex; align-items: center; justify-content: center; background: #000; overflow: hidden; height: 200px;">
                ${mainPreviewThumb
            ? `<img src="${mainPreviewThumb}" style="width: 100%; height: 100%; object-fit: contain;">`
            : `<div style="color: #666;">Video Preview</div>`}
            </div>

            <!-- Verdict Section -->
            <div class="report-verdict" style="margin: 0; padding: 20px; border-color: ${accentColor}; background: ${isFake ? 'rgba(255, 51, 51, 0.1)' : 'rgba(16, 185, 129, 0.1)'}">
                <div class="report-verdict-content" style="width: 100%;">
                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                        <div class="report-verdict-icon" style="background: ${accentColor}; color: white; width: 50px; height: 50px; font-size: 24px;">
                            <i class="fas ${isFake ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
                        </div>
                        <div>
                            <h2 style="color: ${accentColor}; font-size: 24px; margin: 0;">${currentResult.prediction === 'FAKE' ? 'MANIPULATION DETECTED' : 'AUTHENTIC MEDIA'}</h2>
                        </div>
                    </div>
                    
                    <!-- Confidence Meter -->
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px;">
                            <span>Confidence Score</span>
                            <strong>${(currentResult.confidence * 100).toFixed(1)}%</strong>
                        </div>
                        <div style="width: 100%; height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; overflow: hidden;">
                            <div style="width: ${(currentResult.confidence * 100).toFixed(1)}%; height: 100%; background: ${accentColor};"></div>
                        </div>
                    </div>

                    <p style="font-size: 12px; opacity: 0.8; line-height: 1.4;">
                        ${isFake
            ? 'DeepGuard has detected strong indicators of digital manipulation in this video.'
            : 'DeepGuard did not detect any significant anomalies or signs of manipulation.'}
                    </p>
                </div>
            </div>
        </div>

        <div class="report-grid">
            <div class="report-card">
                <h3>ANALYSIS STATISTICS</h3>
                <div class="report-stats-grid">
                    <div class="report-stat-item">
                        <span class="report-stat-label">Duration</span>
                        <span class="report-stat-value">${currentResult.duration.toFixed(1)}s</span>
                    </div>
                    <div class="report-stat-item">
                        <span class="report-stat-label">Frames Scanned</span>
                        <span class="report-stat-value">${currentResult.processed_frames}</span>
                    </div>
                    <div class="report-stat-item">
                        <span class="report-stat-label">Suspicious Frames</span>
                        <span class="report-stat-value" style="color: ${isFake ? '#ff3333' : 'inherit'}">${currentResult.suspicious_frames.length}</span>
                    </div>
                    <div class="report-stat-item">
                        <span class="report-stat-label">Avg. Anomaly Score</span>
                        <span class="report-stat-value">${(currentResult.avg_fake_prob * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>

            <div class="report-card">
                <h3>DETAILED FINDINGS</h3>
                <div style="font-size: 14px; line-height: 1.6; opacity: 0.9;">
                    ${document.getElementById('analysisNotes').innerHTML}
                </div>
            </div>
        </div>

        <div class="report-card" style="margin-bottom: 30px;">
            <h3>TEMPORAL ANALYSIS</h3>
            <img src="${chartImg}" style="width: 100%; height: auto; max-height: 250px; object-fit: contain; margin-top: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
        </div>

        <div class="report-card">
            <h3>KEY FRAMES</h3>
            <div class="report-frames">
                ${framesHTML}
            </div>
        </div>

        <div class="report-footer">
            <p>DeepGuard AI Analysis System • Deepfake Detection Report • ${dateStr}</p>
        </div>
    `;

    // 2. Generate PDF using html2canvas and jspdf
    // Unhide container temporarily (off-screen but rendered)
    reportContainer.style.opacity = '1';
    reportContainer.style.zIndex = '9999';
    reportContainer.style.background = '#0a0a0a';

    try {
        // Ensure libraries are loaded
        if (!window.html2canvas) {
            throw new Error("html2canvas library not loaded");
        }

        // Check for jspdf in various likely locations
        const jsPDF = window.jspdf?.jsPDF || window.jsPDF;
        if (!jsPDF) {
            console.error("jspdf debug:", window.jspdf);
            throw new Error("jspdf library not loaded properly");
        }

        const canvas = await html2canvas(reportContainer, {
            scale: 2, // Improve quality
            useCORS: true,
            backgroundColor: '#0a0a0a',
            logging: false
        });

        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF({
            orientation: 'portrait',
            unit: 'mm',
            format: 'a4'
        });

        const imgWidth = 210; // A4 width in mm
        const pageHeight = 297; // A4 height in mm
        const imgHeight = (canvas.height * imgWidth) / canvas.width;

        pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);

        // If content is longer than one page (unlikely with this layout but good to handle)
        // For now, simpler single page strictly controlled by layout

        pdf.save(`DeepGuard_Report_${Date.now()}.pdf`);

    } finally {
        // Hide again
        reportContainer.style.opacity = '0';
        reportContainer.style.zIndex = '-1000';
    }
}
