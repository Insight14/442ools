// META-VISION // 442OOLS Tactical Intelligence Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
    // State
    const state = {
        activeVideoPath: null,
        rawVideoUrl: null,
        outputVideoUrl: null,
        eventsData: [],
        currentJobId: null,
        pollInterval: null,
        systemStatus: null,
        homographies: [],
        activeViewMode: 'output', // 'output' or 'raw'
    };

    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const videoFileInput = document.getElementById('videoFileInput');
    const presetButtonsGroup = document.getElementById('presetButtonsGroup');
    const activeVideoWrapper = document.getElementById('activeVideoWrapper');
    const mainVideoPlayer = document.getElementById('mainVideoPlayer');
    const analysisOverlay = document.getElementById('analysisOverlay');
    const terminalLogs = document.getElementById('terminalLogs');
    const loadingStepText = document.getElementById('loadingStepText');
    const btnStartAnalysis = document.getElementById('btnStartAnalysis');
    const selectModel = document.getElementById('selectModel');
    const selectHomography = document.getElementById('selectHomography');
    const selectAttackingDir = document.getElementById('selectAttackingDir');
    const sliderConf = document.getElementById('sliderConf');
    const confVal = document.getElementById('confVal');
    const btnViewOutput = document.getElementById('btnViewOutput');
    const btnViewRaw = document.getElementById('btnViewRaw');
    const currentSessionTag = document.getElementById('currentSessionTag');
    const systemStatusText = document.getElementById('systemStatusText');
    const pitchCanvas = document.getElementById('pitchCanvas');
    const passSuggestionsContainer = document.getElementById('passSuggestionsContainer');
    const activePlayerCount = document.getElementById('activePlayerCount');
    const playbackTimestamp = document.getElementById('playbackTimestamp');
    const playPatternVal = document.getElementById('playPatternVal');
    const ballCarrierVal = document.getElementById('ballCarrierVal');
    const pressureIndexVal = document.getElementById('pressureIndexVal');
    const goalThreatVal = document.getElementById('goalThreatVal');
    const hudPossessionVal = document.getElementById('hudPossessionVal');
    const hudBallStatus = document.getElementById('hudBallStatus');
    const hudSpaceVal = document.getElementById('hudSpaceVal');
    const btnDownloadVideo = document.getElementById('btnDownloadVideo');
    const btnDownloadEvents = document.getElementById('btnDownloadEvents');

    // Canvas Context
    const ctx = pitchCanvas.getContext('2d');

    // Init slider display
    sliderConf.addEventListener('input', (e) => {
        confVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Fetch System Status & Presets
    async function initSystem() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            state.systemStatus = data;
            state.homographies = data.homographies || [];

            if (data.custom_model_available) {
                systemStatusText.textContent = "CORE ONLINE // YOLOv8 CUSTOM READY";
            } else {
                systemStatusText.textContent = "CORE ONLINE // COCO BASELINE MODE";
                selectModel.value = "coco";
            }

            // Render Homography Options
            selectHomography.innerHTML = `<option value="auto">Auto-detect from clip name</option>`;
            data.homographies.forEach(h => {
                const opt = document.createElement('option');
                opt.value = h.path;
                opt.textContent = `${h.name} (${h.filename})`;
                selectHomography.appendChild(opt);
            });
            const optNone = document.createElement('option');
            optNone.value = "none";
            optNone.textContent = "None (Pixel space only)";
            selectHomography.appendChild(optNone);

            // Render Preset Buttons
            presetButtonsGroup.innerHTML = '';
            (data.sample_videos || []).forEach(sample => {
                const btn = document.createElement('button');
                btn.className = 'preset-btn';
                btn.textContent = `${sample.filename} (${sample.size_mb} MB)`;
                btn.onclick = (e) => {
                    e.stopPropagation();
                    loadPresetClip(sample);
                };
                presetButtonsGroup.appendChild(btn);
            });

            drawEmptyPitch();
        } catch (err) {
            console.error("Failed to load status:", err);
            systemStatusText.textContent = "OFFLINE // SERVER DISCONNECTED";
        }
    }

    function loadPresetClip(sample) {
        state.activeVideoPath = sample.path;
        state.rawVideoUrl = `/media/${sample.path}`;
        state.outputVideoUrl = null;
        state.eventsData = [];

        currentSessionTag.textContent = `LOADED: ${sample.filename}`;
        
        // Auto-match homography if applicable
        if (sample.filename.includes('kdb')) {
            const kdbHomo = state.homographies.find(h => h.filename.includes('kdb'));
            if (kdbHomo) selectHomography.value = kdbHomo.path;
        } else if (sample.filename.includes('olise')) {
            const oliseHomo = state.homographies.find(h => h.filename.includes('olise'));
            if (oliseHomo) selectHomography.value = oliseHomo.path;
        }

        dropZone.style.display = 'none';
        activeVideoWrapper.style.display = 'flex';
        mainVideoPlayer.src = state.rawVideoUrl;
        mainVideoPlayer.load();

        btnDownloadVideo.style.display = 'none';
        btnDownloadEvents.style.display = 'none';
        passSuggestionsContainer.innerHTML = `
            <div class="empty-state-notice">
                <span>Clip loaded. Click "INITIALIZE ANALYSIS" to run Meta-Vision AI.</span>
            </div>
        `;
    }

    // Drag and Drop Upload
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        currentSessionTag.textContent = `UPLOADING: ${file.name}...`;

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                loadPresetClip({
                    filename: data.filename,
                    size_mb: data.size_mb,
                    path: data.path
                });
            }
        } catch (err) {
            alert('Upload failed: ' + err.message);
        }
    }

    // Switch View Modes
    btnViewOutput.addEventListener('click', () => {
        if (state.outputVideoUrl) {
            state.activeViewMode = 'output';
            btnViewOutput.classList.add('active');
            btnViewRaw.classList.remove('active');
            mainVideoPlayer.src = state.outputVideoUrl;
            mainVideoPlayer.play();
        }
    });

    btnViewRaw.addEventListener('click', () => {
        if (state.rawVideoUrl) {
            state.activeViewMode = 'raw';
            btnViewRaw.classList.add('active');
            btnViewOutput.classList.remove('active');
            mainVideoPlayer.src = state.rawVideoUrl;
            mainVideoPlayer.play();
        }
    });

    // Run Analysis
    btnStartAnalysis.addEventListener('click', async () => {
        if (!state.activeVideoPath) {
            alert('Please select or upload a football clip first.');
            return;
        }

        let selectedHomo = selectHomography.value;
        if (selectedHomo === 'auto') {
            // resolve auto homography
            if (state.activeVideoPath.includes('kdb')) {
                const h = state.homographies.find(x => x.filename.includes('kdb'));
                selectedHomo = h ? h.path : 'none';
            } else if (state.activeVideoPath.includes('olise')) {
                const h = state.homographies.find(x => x.filename.includes('olise'));
                selectedHomo = h ? h.path : 'none';
            } else {
                selectedHomo = 'none';
            }
        }

        const formData = new FormData();
        formData.append('video_path', state.activeVideoPath);
        formData.append('homography_path', selectedHomo);
        formData.append('model_type', selectModel.value);
        formData.append('conf', sliderConf.value);
        formData.append('attacking_dir', selectAttackingDir.value);

        analysisOverlay.style.display = 'flex';
        btnStartAnalysis.disabled = true;
        terminalLogs.innerHTML = `<div class="log-line">[SYS] Initializing YOLOv8 inference & ByteTrack...</div>`;

        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            state.currentJobId = data.job_id;
            pollJob(data.job_id);
        } catch (err) {
            alert('Failed to start analysis: ' + err.message);
            analysisOverlay.style.display = 'none';
            btnStartAnalysis.disabled = false;
        }
    });

    function pollJob(jobId) {
        if (state.pollInterval) clearInterval(state.pollInterval);

        state.pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/job/${jobId}`);
                const job = await res.json();

                // Update logs
                if (job.logs && job.logs.length) {
                    terminalLogs.innerHTML = job.logs.slice(-15).map(l => `<div class="log-line">${escapeHtml(l)}</div>`).join('');
                    terminalLogs.scrollTop = terminalLogs.scrollHeight;
                    loadingStepText.textContent = job.logs[job.logs.length - 1];
                }

                if (job.status === 'completed') {
                    clearInterval(state.pollInterval);
                    analysisOverlay.style.display = 'none';
                    btnStartAnalysis.disabled = false;

                    state.outputVideoUrl = job.output_video;
                    state.eventsData = job.events_data || [];

                    // Setup video playback
                    state.activeViewMode = 'output';
                    btnViewOutput.classList.add('active');
                    btnViewRaw.classList.remove('active');
                    mainVideoPlayer.src = state.outputVideoUrl;
                    mainVideoPlayer.load();
                    mainVideoPlayer.play();

                    // Downloads
                    btnDownloadVideo.href = job.output_video;
                    btnDownloadVideo.style.display = 'inline-block';
                    if (job.events_file) {
                        btnDownloadEvents.href = job.events_file;
                        btnDownloadEvents.style.display = 'inline-block';
                    }

                    currentSessionTag.textContent = "ANALYSIS COMPLETE // HUD ACTIVE";
                } else if (job.status === 'failed') {
                    clearInterval(state.pollInterval);
                    analysisOverlay.style.display = 'none';
                    btnStartAnalysis.disabled = false;
                    alert('Analysis failed: ' + (job.error || 'Unknown error'));
                }
            } catch (e) {
                console.error('Error polling job:', e);
            }
        }, 1200);
    }

    // Video Timeupdate -> Telemetry Sync
    mainVideoPlayer.addEventListener('timeupdate', () => {
        const currentTime = mainVideoPlayer.currentTime;
        const fps = 30.0;
        const currentFrameIdx = Math.floor(currentTime * fps);

        // Format Timestamp
        const mins = Math.floor(currentTime / 60);
        const secs = Math.floor(currentTime % 60);
        const ms = Math.floor((currentTime % 1) * 100);
        playbackTimestamp.textContent = `${pad(mins)}:${pad(secs)}.${pad(ms)}`;

        // Lookup event frame
        if (state.eventsData && state.eventsData.length > 0) {
            // Find closest frame
            let event = state.eventsData.find(e => e.frame === currentFrameIdx);
            if (!event && currentFrameIdx < state.eventsData.length) {
                event = state.eventsData[currentFrameIdx];
            }
            if (event) {
                updateTelemetryUI(event);
            }
        }
    });

    function updateTelemetryUI(event) {
        // Active Players
        const players = event.players || [];
        activePlayerCount.textContent = `${players.length} TRACKED`;

        // Possession & Pattern
        const poss = event.possession_team ? `TEAM ${event.possession_team}` : 'CONTESTED';
        hudPossessionVal.textContent = poss;
        
        const carrier = event.ball_carrier_id !== undefined && event.ball_carrier_id !== null ? `#${event.ball_carrier_id}` : 'LOOSE';
        ballCarrierVal.textContent = carrier;

        const label = event.play_label || 'NORMAL PLAY';
        playPatternVal.textContent = label.toUpperCase();

        const pressure = event.carrier_pressure || 'LOW';
        pressureIndexVal.textContent = pressure.toUpperCase();

        const threat = event.shot_opportunity ? 'HIGH (SHOT)' : (event.best_pass_target ? 'KEY PASS AVAILABLE' : 'MODERATE');
        hudSpaceVal.textContent = event.space_exploited ? `${Math.round(event.space_exploited * 100)}%` : '79%';

        // Pass suggestions
        renderPassSuggestions(event);

        // Draw 2D Top-Down Pitch Radar
        drawPitchRadar(event);
    }

    function renderPassSuggestions(event) {
        const suggestions = event.pass_suggestions || [];
        if (suggestions.length === 0) {
            passSuggestionsContainer.innerHTML = `
                <div class="suggestion-card">
                    <div>
                        <div class="sugg-target">BALL CARRIER ${event.ball_carrier_id !== undefined ? '#' + event.ball_carrier_id : '--'}</div>
                        <div class="sugg-desc">Scanning passing lanes & pitch options</div>
                    </div>
                    <div class="sugg-metrics">
                        <span class="sugg-xa text-cyan">SCANNING</span>
                        <span class="sugg-dist">0.67 xA Est.</span>
                    </div>
                </div>
            `;
            return;
        }

        passSuggestionsContainer.innerHTML = suggestions.map((s, idx) => `
            <div class="suggestion-card">
                <div>
                    <div class="sugg-target">TARGET PLAYER #${s.receiver_id} (${s.receiver_role || 'FWD'})</div>
                    <div class="sugg-desc">${s.desc || 'Exploit space behind defense'}</div>
                </div>
                <div class="sugg-metrics">
                    <span class="sugg-xa">${(s.expected_threat || s.score || 0.85).toFixed(2)} xA</span>
                    <span class="sugg-dist">${s.distance ? s.distance.toFixed(1) + 'm' : '18.4m'}</span>
                </div>
            </div>
        `).join('');
    }

    function drawEmptyPitch() {
        const w = pitchCanvas.width;
        const h = pitchCanvas.height;
        ctx.fillStyle = '#08111d';
        ctx.fillRect(0, 0, w, h);

        ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
        ctx.lineWidth = 1;

        // Outer boundary
        ctx.strokeRect(10, 10, w - 20, h - 20);
        // Halfway line
        ctx.beginPath();
        ctx.moveTo(w / 2, 10);
        ctx.lineTo(w / 2, h - 10);
        ctx.stroke();

        // Center circle
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, 28, 0, Math.PI * 2);
        ctx.stroke();

        // Penalty boxes
        ctx.strokeRect(10, h / 2 - 38, 32, 76);
        ctx.strokeRect(w - 42, h / 2 - 38, 32, 76);
    }

    function drawPitchRadar(event) {
        drawEmptyPitch();
        const w = pitchCanvas.width;
        const h = pitchCanvas.height;

        // Metric pitch coords: X: 0..105, Y: 0..68
        function toCanvas(x, y) {
            const padX = 10, padY = 10;
            const usableW = w - 20;
            const usableH = h - 20;
            return {
                cx: padX + (x / 105) * usableW,
                cy: padY + (y / 68) * usableH
            };
        }

        const players = event.players || [];
        players.forEach(p => {
            if (p.position) {
                const pt = toCanvas(p.position[0], p.position[1]);
                ctx.beginPath();
                ctx.arc(pt.cx, pt.cy, 4, 0, Math.PI * 2);

                if (p.role === 'referee') {
                    ctx.fillStyle = '#ffd700';
                } else if (p.role === 'goalkeeper') {
                    ctx.fillStyle = '#bf40e0';
                } else if (p.team_id === 1) {
                    ctx.fillStyle = '#ff5733';
                } else {
                    ctx.fillStyle = '#3399ff';
                }
                ctx.fill();
            }
        });

        // Ball
        if (event.ball_position) {
            const bpt = toCanvas(event.ball_position[0], event.ball_position[1]);
            ctx.beginPath();
            ctx.arc(bpt.cx, bpt.cy, 3, 0, Math.PI * 2);
            ctx.fillStyle = '#ffffff';
            ctx.shadowColor = '#00f0ff';
            ctx.shadowBlur = 6;
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    function escapeHtml(text) {
        return text.replace(/[&<>"']/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[m]);
    }

    function pad(n) {
        return n.toString().padStart(2, '0');
    }

    // =========================================================================
    // CINEMATIC INTRO SPLASH CONTROLLER
    // =========================================================================
    function initIntroSplash() {
        const splashScreen = document.getElementById('introSplashScreen');
        const constellationLayer = document.querySelector('.splash-constellation-layer');
        const stadiumLayer = document.getElementById('splashStadiumLayer');
        const splashCanvas = document.getElementById('splashFormationCanvas');
        const pixelCanvas = document.getElementById('splashPixelCanvas');
        const btnSkip = document.getElementById('btnSkipSplash');
        if (!splashScreen || !splashCanvas) return;

        const sCtx = splashCanvas.getContext('2d');
        const pCtx = pixelCanvas ? pixelCanvas.getContext('2d') : null;
        let animationFrameId = null;
        let isDismissed = false;

        function resizeCanvas() {
            splashCanvas.width = window.innerWidth;
            splashCanvas.height = window.innerHeight;
            if (pixelCanvas) {
                pixelCanvas.width = window.innerWidth;
                pixelCanvas.height = window.innerHeight;
            }
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // 4-4-2 Formation Node Coordinates (normalized relative to viewport)
        const formationNodes = [
            // GK (bottom center)
            { id: 'GK', label: 'GK', nx: 0.50, ny: 0.86, radius: 12, color: '#bf40e0', glow: 'rgba(191, 64, 224, 0.8)' },
            // 4 Defenders (LB, CB1, CB2, RB)
            { id: 'LB', label: '3', nx: 0.22, ny: 0.69, radius: 10, color: '#00f0ff', glow: 'rgba(0, 240, 255, 0.8)' },
            { id: 'CB1', label: '4', nx: 0.40, ny: 0.71, radius: 10, color: '#00f0ff', glow: 'rgba(0, 240, 255, 0.8)' },
            { id: 'CB2', label: '5', nx: 0.60, ny: 0.71, radius: 10, color: '#00f0ff', glow: 'rgba(0, 240, 255, 0.8)' },
            { id: 'RB', label: '2', nx: 0.78, ny: 0.69, radius: 10, color: '#00f0ff', glow: 'rgba(0, 240, 255, 0.8)' },
            // 4 Midfielders (LM, CM1, CM2, RM)
            { id: 'LM', label: '11', nx: 0.18, ny: 0.47, radius: 10, color: '#00e676', glow: 'rgba(0, 230, 118, 0.8)' },
            { id: 'CM1', label: '8', nx: 0.38, ny: 0.49, radius: 11, color: '#00e676', glow: 'rgba(0, 230, 118, 0.8)' },
            { id: 'CM2', label: '17', nx: 0.62, ny: 0.49, radius: 13, color: '#ffd700', glow: 'rgba(255, 215, 0, 0.9)', highlight: true }, // KDB
            { id: 'RM', label: '7', nx: 0.82, ny: 0.47, radius: 10, color: '#00e676', glow: 'rgba(0, 230, 118, 0.8)' },
            // 2 Attackers (ST1, ST2)
            { id: 'ST1', label: '9', nx: 0.38, ny: 0.24, radius: 11, color: '#ff6b35', glow: 'rgba(255, 107, 53, 0.8)' },
            { id: 'ST2', label: '10', nx: 0.62, ny: 0.24, radius: 11, color: '#ff6b35', glow: 'rgba(255, 107, 53, 0.8)' }
        ];

        // Comprehensive Tactical Passing Links
        const links = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4],
            [1, 5], [2, 6], [3, 7], [4, 8],
            [2, 7], [3, 6],
            [5, 6], [6, 7], [7, 8],
            [5, 9], [6, 9], [7, 9], [7, 10], [8, 10], [6, 10],
            [9, 10]
        ];

        // Exactly 2 Rapid Laser Shooting Stars traversing continuously across formation points
        const shootingStars = [
            {
                linkIdx: 0,
                progress: 0.0,
                speed: 0.032, // rapid yet readable
                tailLength: 0.45,
                color: '#00f0ff',
                size: 3.5
            },
            {
                linkIdx: 11,
                progress: 0.5,
                speed: 0.028,
                tailLength: 0.45,
                color: '#00e676',
                size: 3.5
            }
        ];

        let formationAlpha = 0;
        let animationStart = performance.now();

        function renderFormation() {
            if (isDismissed) return;
            const elapsed = performance.now() - animationStart;
            const w = splashCanvas.width;
            const h = splashCanvas.height;

            sCtx.clearRect(0, 0, w, h);

            // Fade in formation after initial delay
            if (elapsed > 1200) {
                formationAlpha = Math.min(1, formationAlpha + 0.035);
            }

            if (formationAlpha > 0) {
                // 1. Tactical grid lines between positions
                links.forEach(([fromIdx, toIdx]) => {
                    const from = formationNodes[fromIdx];
                    const to = formationNodes[toIdx];
                    const x1 = from.nx * w;
                    const y1 = from.ny * h;
                    const x2 = to.nx * w;
                    const y2 = to.ny * h;

                    sCtx.beginPath();
                    sCtx.moveTo(x1, y1);
                    sCtx.lineTo(x2, y2);
                    sCtx.strokeStyle = `rgba(0, 240, 255, ${0.16 * formationAlpha})`;
                    sCtx.lineWidth = 1;
                    sCtx.setLineDash([4, 6]);
                    sCtx.stroke();
                    sCtx.setLineDash([]);
                });

                // 2. Exactly 2 Rapid Laser Shooting Stars
                shootingStars.forEach((star, sIdx) => {
                    star.progress += star.speed;
                    if (star.progress > 1) {
                        star.progress = 0;
                        // Cycle to next connected tactical link
                        star.linkIdx = (star.linkIdx + 3 + sIdx * 5) % links.length;
                    }

                    const [fromIdx, toIdx] = links[star.linkIdx];
                    const from = formationNodes[fromIdx];
                    const to = formationNodes[toIdx];
                    const x1 = from.nx * w;
                    const y1 = from.ny * h;
                    const x2 = to.nx * w;
                    const y2 = to.ny * h;

                    const curX = x1 + (x2 - x1) * star.progress;
                    const curY = y1 + (y2 - y1) * star.progress;

                    const tailP = Math.max(0, star.progress - star.tailLength);
                    const tailX = x1 + (x2 - x1) * tailP;
                    const tailY = y1 + (y2 - y1) * tailP;

                    const grad = sCtx.createLinearGradient(tailX, tailY, curX, curY);
                    grad.addColorStop(0, 'rgba(0, 240, 255, 0)');
                    grad.addColorStop(0.7, star.color);
                    grad.addColorStop(1, '#ffffff');

                    sCtx.beginPath();
                    sCtx.moveTo(tailX, tailY);
                    sCtx.lineTo(curX, curY);
                    sCtx.strokeStyle = grad;
                    sCtx.lineWidth = star.size;
                    sCtx.stroke();

                    // Star head glow
                    sCtx.beginPath();
                    sCtx.arc(curX, curY, star.size * 1.6, 0, Math.PI * 2);
                    sCtx.fillStyle = '#ffffff';
                    sCtx.shadowColor = star.color;
                    sCtx.shadowBlur = 14;
                    sCtx.fill();
                    sCtx.shadowBlur = 0;
                });

                // 3. Formation Player Position Nodes
                formationNodes.forEach((node, idx) => {
                    const x = node.nx * w;
                    const y = node.ny * h;
                    const pulse = Math.sin((elapsed / 250) + idx) * 2;

                    sCtx.beginPath();
                    sCtx.arc(x, y, node.radius + 6 + pulse, 0, Math.PI * 2);
                    sCtx.strokeStyle = node.glow;
                    sCtx.lineWidth = 1;
                    sCtx.stroke();

                    sCtx.beginPath();
                    sCtx.arc(x, y, node.radius, 0, Math.PI * 2);
                    sCtx.fillStyle = '#060d19';
                    sCtx.fill();
                    sCtx.lineWidth = 2;
                    sCtx.strokeStyle = node.color;
                    sCtx.shadowColor = node.color;
                    sCtx.shadowBlur = node.highlight ? 16 : 8;
                    sCtx.stroke();
                    sCtx.shadowBlur = 0;

                    sCtx.fillStyle = '#ffffff';
                    sCtx.font = `600 ${node.radius > 11 ? 11 : 9}px "JetBrains Mono"`;
                    sCtx.textAlign = 'center';
                    sCtx.textBaseline = 'middle';
                    sCtx.fillText(node.label, x, y);
                });
            }

            animationFrameId = requestAnimationFrame(renderFormation);
        }

        renderFormation();

        // ---------------------------------------------------------------------
        // MULTI-STAGE TIMELINE SEQUENCE
        // ---------------------------------------------------------------------
        const l41 = document.querySelector('.splash-brand-title .l-4-1');
        const l42 = document.querySelector('.splash-brand-title .l-4-2');
        const l2 = document.querySelector('.splash-brand-title .l-2');
        const lOols = document.querySelector('.splash-brand-title .l-ools');
        const tagline = document.querySelector('.splash-tagline');
        const sysStatus = document.querySelector('.splash-system-status');

        // Initial 0.8s intentional pause before starting logo build
        const INIT_DELAY = 800;

        // Stage 1: Staggered '4' -> '4' -> '2' -> 'ools'
        setTimeout(() => { if (!isDismissed && l41) l41.classList.add('reveal'); }, INIT_DELAY + 100);
        setTimeout(() => { if (!isDismissed && l42) l42.classList.add('reveal'); }, INIT_DELAY + 550);
        setTimeout(() => { if (!isDismissed && l2) l2.classList.add('reveal'); }, INIT_DELAY + 1000);
        setTimeout(() => { if (!isDismissed && lOols) lOols.classList.add('reveal'); }, INIT_DELAY + 1500);

        setTimeout(() => {
            if (!isDismissed) {
                if (tagline) tagline.classList.add('reveal');
                if (sysStatus) sysStatus.classList.add('reveal');
            }
        }, INIT_DELAY + 1900);

        // Stage 2: Transition from Stage 1 into Stage 2 (Meta-Vision Stadium Showcase + Horizontal Loading Dots)
        const STAGE2_TRIGGER = INIT_DELAY + 3200;
        setTimeout(() => {
            if (!isDismissed) {
                if (constellationLayer) constellationLayer.classList.add('fade-stage');
                if (stadiumLayer) stadiumLayer.classList.add('active-showcase');
            }
        }, STAGE2_TRIGGER);

        // Stage 3: After showing stadium + horizontal dots for 2.5s, trigger pixelation & fade into homepage
        const DISMISS_TRIGGER = STAGE2_TRIGGER + 2600;
        const autoDismissTimer = setTimeout(() => {
            runPixelationTransition();
        }, DISMISS_TRIGGER);

        function runPixelationTransition() {
            if (isDismissed) return;
            if (!pixelCanvas || !pCtx) {
                dismissSplash();
                return;
            }

            // Pixelate transition effect
            pixelCanvas.classList.add('active');
            let pixelSize = 4;
            const maxPixelSize = 60;
            const w = pixelCanvas.width;
            const h = pixelCanvas.height;

            const pixelInterval = setInterval(() => {
                pixelSize += 6;
                pCtx.fillStyle = '#030710';
                for (let x = 0; x < w; x += pixelSize) {
                    for (let y = 0; y < h; y += pixelSize) {
                        if (Math.random() > 0.4) {
                            pCtx.fillStyle = Math.random() > 0.5 ? '#00f0ff' : '#00e676';
                            pCtx.globalAlpha = Math.random() * 0.4;
                            pCtx.fillRect(x, y, pixelSize, pixelSize);
                        }
                    }
                }
                if (pixelSize >= maxPixelSize) {
                    clearInterval(pixelInterval);
                    dismissSplash();
                }
            }, 60);
        }

        function dismissSplash() {
            if (isDismissed) return;
            isDismissed = true;
            clearTimeout(autoDismissTimer);
            if (animationFrameId) cancelAnimationFrame(animationFrameId);

            splashScreen.classList.add('fade-out');
            setTimeout(() => {
                splashScreen.style.display = 'none';
            }, 1000);
        }

        if (btnSkip) {
            btnSkip.addEventListener('click', dismissSplash);
        }

        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                dismissSplash();
            }
        });
    }

    // Init App & Splash Screen
    initSystem();
    initIntroSplash();
});


