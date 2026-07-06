// ========== STATE MANAGEMENT ==========
const params = new URLSearchParams(window.location.search);
const brokerParam = params.get('broker');
const wssParam = params.get('wss');
const ipParam = params.get('ip');

const defaultBroker = "smart-surveillance-broker.onrender.com";

let initialWsAddress;
if (brokerParam) {
    initialWsAddress = `wss://${brokerParam}?role=frontend`;
} else if (wssParam) {
    initialWsAddress = `wss://${wssParam}`;
} else if (ipParam) {
    initialWsAddress = `ws://${ipParam}:8765`;
} else {
    initialWsAddress = `wss://${defaultBroker}?role=frontend`;
}

let state = {
    connected: false,
    paused: false,
    stopped: false,
    websocket: null,
    charts: {},
    reconnectTimer: null,
    wsAddress: initialWsAddress,
    lastPacket: 'waiting...',
    sessionData: null
};

// ========== CHART INITIALIZATION ==========
function initializeChart(labels = [], personsData = [], objectsData = []) {
    const canvas = document.getElementById('activityChart');
    if (!canvas) return;

    if (state.charts['activityChart']) {
        state.charts['activityChart'].destroy();
    }

    const ctx = canvas.getContext('2d');
    state.charts['activityChart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels.length ? labels : ['0s'],
            datasets: [
                {
                    label: 'Persons',
                    data: personsData.length ? personsData : [0],
                    borderColor: '#C97C2C', // Copper
                    backgroundColor: 'rgba(201, 124, 44, 0.05)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Tracked Assets',
                    data: objectsData.length ? objectsData : [0],
                    borderColor: '#2ECC71', // Emerald Green
                    backgroundColor: 'rgba(46, 204, 113, 0.05)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                        color: '#A7ADB5'
                    },
                    grid: {
                        color: 'rgba(49, 54, 63, 0.3)'
                    }
                },
                x: {
                    ticks: {
                        color: '#A7ADB5'
                    },
                    grid: {
                        color: 'rgba(49, 54, 63, 0.3)'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#F5F5F5'
                    }
                }
            }
        }
    });
}

// ========== ATTACH EVENTS ==========
document.addEventListener('DOMContentLoaded', () => {
    initializeChart();
    attachEventListeners();

    const params = new URLSearchParams(window.location.search);
    const brokerParam = params.get('broker');
    const wssParam = params.get('wss');

    if (brokerParam || wssParam) {
        console.log("Connecting to WebSocket from query param:", state.wsAddress);
        connectWebSocket();
    } else {
        // Fetch session.json to check if there is an active WebSocket override URL
        fetch('session.json')
            .then(res => res.json())
            .then(data => {
                if (data && data.ws_url) {
                    state.wsAddress = data.ws_url.includes('?') ? data.ws_url : `${data.ws_url}?role=frontend`;
                    console.log("Connecting to WebSocket from session.json:", state.wsAddress);
                }
                connectWebSocket();
            })
            .catch(err => {
                console.log("Connecting using default WS address:", state.wsAddress);
                connectWebSocket();
            });
    }

    // Check if 'session' query param is present for auto-loading
    const sessionParam = params.get('session');
    if (sessionParam) {
        document.getElementById('loaded-file-name').textContent = `Loading: ${sessionParam}...`;
        fetch(sessionParam)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                state.sessionData = data;
                populateSurveillanceDashboard(data);
                document.getElementById('loaded-file-name').textContent = `Auto-loaded: ${sessionParam}`;
            })
            .catch(err => {
                document.getElementById('loaded-file-name').textContent = `Failed to load session`;
                console.error('Failed to auto-load session JSON:', err);
            });
    }
});

function attachEventListeners() {
    // Session JSON Loader
    const uploader = document.getElementById('session-upload-input');
    if (uploader) {
        uploader.addEventListener('change', handleSessionUpload);
    }

    // Toggle live updates
    const toggleBtn = document.getElementById('toggle-updates');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            state.paused = !state.paused;
            toggleBtn.textContent = state.paused ? 'Resume Live' : 'Pause Live';
            toggleBtn.className = state.paused 
                ? 'flex-grow px-4 py-3 bg-[#2ECC71] text-[#0F1115] rounded-lg transition font-bold text-xs whitespace-nowrap'
                : 'flex-grow px-4 py-3 bg-[#1A1E23] hover:bg-[#31363F] text-[#F5F5F5] border border-[#31363F] rounded-lg transition font-semibold text-xs whitespace-nowrap';
            
            if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
                state.websocket.send(state.paused ? "PAUSE" : "RESUME");
            }
        });
    }

    // Stop updates/processing
    const stopBtn = document.getElementById('stop-updates');
    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            if (confirm("Are you sure you want to stop the live CCTV processing? This will trigger final report generation and end the session.")) {
                state.stopped = true;
                if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
                    state.websocket.send("STOP");
                }
                stopBtn.disabled = true;
                stopBtn.textContent = "Stopping...";
                stopBtn.className = "flex-grow px-4 py-3 bg-[#31363F] text-[#A7ADB5] rounded-lg transition font-semibold text-xs whitespace-nowrap cursor-not-allowed";
                if (toggleBtn) {
                    toggleBtn.disabled = true;
                    toggleBtn.className = "flex-grow px-4 py-3 bg-[#31363F] text-[#A7ADB5] rounded-lg transition font-semibold text-xs whitespace-nowrap cursor-not-allowed";
                }
            }
        });
    }

    // CSV Export
    const exportBtn = document.getElementById('export-data');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportSessionCSV);
    }
}

// ========== HANDLE UPLOAD ==========
function handleSessionUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    document.getElementById('loaded-file-name').textContent = `Loaded: ${file.name}`;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);
            state.sessionData = data;
            populateSurveillanceDashboard(data);
        } catch (err) {
            alert('Error parsing JSON session file: ' + err.message);
            console.error(err);
        }
    };
    reader.readAsText(file);
}

// ========== POPULATE DASHBOARD ==========
function populateSurveillanceDashboard(data) {
    // 1. Session Status / Threat level
    const statusBadge = document.getElementById('session-status-badge');
    const statusDesc = document.getElementById('session-status-desc');
    const statusBanner = document.getElementById('status-banner');
    const bannerText = document.getElementById('banner-text');

    const status = (data.status || 'NORMAL').toUpperCase();
    statusBadge.textContent = status;

    if (status === 'THEFT') {
        statusBadge.className = 'text-2xl font-black text-[#E63946] tracking-wider glow-red';
        statusDesc.textContent = 'CRITICAL: Verified theft incident confirmed!';
        statusBanner.className = 'alert-banner alert-critical';
        bannerText.textContent = 'CRITICAL ALERT - ACTIVE THEFT CONFIRMED';
    } else if (status === 'SUSPICIOUS') {
        statusBadge.className = 'text-2xl font-black text-[#F39C12] tracking-wider glow-yellow';
        statusDesc.textContent = 'WARNING: Suspicious object movement or activity detected';
        statusBanner.className = 'alert-banner alert-warning';
        bannerText.textContent = 'WARNING - SUSPICIOUS BEHAVIOUR DETECTED';
    } else {
        statusBadge.className = 'text-2xl font-black text-[#2ECC71] tracking-wider';
        statusDesc.textContent = 'No suspicious activity detected';
        statusBanner.className = 'alert-banner alert-normal';
        bannerText.textContent = 'Surveillance Active - System Monitoring Healthy';
    }

    // 2. Metrics Cards
    document.getElementById('unique-persons-count').textContent = data.unique_persons || 0;
    document.getElementById('total-frames-count').textContent = data.total_frames || 0;
    document.getElementById('avg-confidence').textContent = data.avg_confidence || '0.0%';
    document.getElementById('qwen-summary-text').textContent = data.summary || 'No summary text available.';
    document.getElementById('session-timestamp').textContent = `Session: ${data.timestamp || '--'}`;

    // 3. Suspect Database Grid
    const suspectsGrid = document.getElementById('suspects-grid');
    const suspectsCountBadge = document.getElementById('suspects-count');
    suspectsGrid.innerHTML = '';

    const suspects = data.suspects || [];
    suspectsCountBadge.textContent = `${suspects.length} Suspect${suspects.length !== 1 ? 's' : ''}`;

    if (suspects.length) {
        suspects.forEach(suspect => {
            const card = document.createElement('div');
            card.id = `suspect-card-${suspect.id}`;
            card.className = 'bg-[#1A1E23] p-3 rounded-lg border border-[#E63946] border-opacity-35 flex flex-col justify-between hover:border-[#E63946] transition';
            
            const imageSrc = suspect.photo;
            
            card.innerHTML = `
                <div class="suspect-img-container mb-2 h-28 flex items-center justify-center bg-black bg-opacity-40">
                    <img src="${imageSrc}" onerror="this.src='https://placehold.co/150x200/222/ef4444?text=Photo'" class="max-h-full max-w-full object-contain cursor-pointer rounded" onclick="openImageModal('${imageSrc}', 'Suspect Person ${suspect.id}')">
                </div>
                <div>
                    <h3 class="font-bold text-[#E63946] text-xs">Suspect ID ${suspect.id}</h3>
                    <p class="text-[10px] text-[#A7ADB5] mt-0.5">Time: ${suspect.timestamp || '--'}</p>
                    <p class="text-[10px] text-[#F5F5F5] mt-1 bg-[#E63946] bg-opacity-10 p-1.5 rounded leading-tight border border-[#E63946] border-opacity-20">
                        ${suspect.details}
                    </p>
                </div>
            `;
            suspectsGrid.appendChild(card);
        });
    } else {
        suspectsGrid.innerHTML = `
            <div class="text-center py-8 text-[#A7ADB5] border border-dashed border-[#31363F] rounded-lg text-xs">
                No suspects captured.
            </div>
        `;
    }

    // 4. Persons ID Database Grid
    const idPhotosContainer = document.getElementById('id-photos-container');
    idPhotosContainer.innerHTML = '';

    let idPhotos = data.id_photos || [];
    // Fallback: extract first appearance from keyframes if id_photos list is missing
    if (!idPhotos.length && data.keyframes) {
        Object.keys(data.keyframes).forEach(id => {
            const crops = data.keyframes[id];
            if (crops && crops.length) {
                idPhotos.push({
                    id: parseInt(id),
                    photo: crops[0].photo
                });
            }
        });
    }

    if (idPhotos.length) {
        idPhotos.forEach(item => {
            const card = document.createElement('div');
            card.id = `id-photo-card-${item.id}`;
            card.className = 'id-photo-card flex flex-col items-center justify-between';
            card.innerHTML = `
                <div class="w-full aspect-square bg-[#0A0C0D] flex items-center justify-center overflow-hidden rounded border border-[#31363F] mb-2">
                    <img src="${item.photo}" onerror="this.src='https://placehold.co/100/222/2ecc71?text=Photo'" class="max-h-full max-w-full object-contain rounded" onclick="openImageModal('${item.photo}', 'Tracked ID ${item.id}')">
                </div>
                <div class="text-xs font-bold text-[#A7ADB5] text-center">ID ${item.id}</div>
            `;
            idPhotosContainer.appendChild(card);
        });
    } else {
        idPhotosContainer.innerHTML = `
            <div class="col-span-full text-center py-8 text-[#A7ADB5] border border-dashed border-[#31363F] rounded-lg text-xs">
                Awaiting tracked individuals database...
            </div>
        `;
    }

    // 5. Chart data
    if (data.chart_data) {
        initializeChart(
            data.chart_data.labels || [],
            data.chart_data.persons || [],
            data.chart_data.objects || []
        );
    } else if (data.events && data.events.length) {
        // Fallback chart: reconstruct timeline indices from events
        const labels = [];
        const persons = [];
        const objects = [];
        let pCount = 0;
        let oCount = 0;
        data.events.forEach((evt, idx) => {
            labels.push(`E${idx + 1}`);
            const lower = evt.toLowerCase();
            if (lower.includes('entered') || lower.includes('in')) pCount++;
            if (lower.includes('exited') || lower.includes('out')) pCount = Math.max(0, pCount - 1);
            if (lower.includes('theft') || lower.includes('picked')) oCount++;
            persons.push(pCount);
            objects.push(oCount);
        });
        initializeChart(labels, persons, objects);
    }
}

// ========== IMAGE MODAL ==========
function openImageModal(src, title) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('image-modal-img');
    const modalTitle = document.getElementById('image-modal-title');

    modalTitle.textContent = title;
    modalImg.src = src;
    modal.classList.remove('hidden');
}

function closeImageModal() {
    const modal = document.getElementById('image-modal');
    modal.classList.add('hidden');
}

// ========== EXPORT TIMELINE CSV ==========
function exportSessionCSV() {
    if (!state.sessionData) {
        alert('Please load a session JSON first.');
        return;
    }

    const events = state.sessionData.events || [];
    const csvContent = [
        ['Surveillance Event Log'],
        ['Timestamp / Log Entry'],
        ...events.map(e => [e])
    ]
    .map(row => `"${row[0].replace(/"/g, '""')}"`)
    .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `surveillance-log-${state.sessionData.timestamp || 'session'}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
}

// ========== WEBSOCKET FOR LIVE FEED ==========
function connectWebSocket() {
    if (state.connected || (state.websocket && state.websocket.readyState === WebSocket.CONNECTING)) {
        return;
    }

    try {
        state.websocket = new WebSocket(state.wsAddress);

        state.websocket.onopen = function () {
            state.connected = true;
            updateConnectionStatus(true);
        };

        state.websocket.onmessage = function (event) {
            updateLastPacketInfo(event.data);
            const data = event.data;
            const delimiter = data.includes('|') ? '|' : ',';
            const type = data.split(delimiter)[0];
            
            if (type === 'FINISHED' || type === 'FINISHED_DATA') {
                state.stopped = true;
                const stopBtn = document.getElementById('stop-updates');
                const toggleBtn = document.getElementById('toggle-updates');
                if (stopBtn) {
                    stopBtn.disabled = true;
                    stopBtn.textContent = "Stopped";
                    stopBtn.className = "flex-grow px-4 py-3 bg-[#31363F] text-[#A7ADB5] rounded-lg transition font-semibold text-xs whitespace-nowrap cursor-not-allowed";
                }
                if (toggleBtn) {
                    toggleBtn.disabled = true;
                    toggleBtn.className = "flex-grow px-4 py-3 bg-[#31363F] text-[#A7ADB5] rounded-lg transition font-semibold text-xs whitespace-nowrap cursor-not-allowed";
                }
            }

            if (!state.paused || type === 'FINISHED' || type === 'FINISHED_DATA') {
                processLiveMessage(event.data);
            }
        };

        state.websocket.onerror = function (error) {
            state.connected = false;
            updateConnectionStatus(false);
            scheduleReconnect();
        };

        state.websocket.onclose = function () {
            state.connected = false;
            updateConnectionStatus(false);
            scheduleReconnect();
        };

    } catch (error) {
        state.connected = false;
        updateConnectionStatus(false);
        scheduleReconnect();
    }
}

function scheduleReconnect() {
    if (state.reconnectTimer || state.connected || state.stopped) return;
    state.reconnectTimer = setTimeout(() => {
        state.reconnectTimer = null;
        connectWebSocket();
    }, 5000);
}

function updateConnectionStatus(connected) {
    const statusText = document.getElementById('status-text');
    const dot = document.getElementById('connection-status');
    if (!statusText || !dot) return;

    if (connected) {
        statusText.textContent = 'Connected (Live)';
        statusText.style.color = '#2ECC71';
        dot.className = 'w-3.5 h-3.5 rounded-full bg-[#2ECC71] animate-pulse';
    } else {
        statusText.textContent = 'Disconnected';
        statusText.style.color = '#E63946';
        dot.className = 'w-3.5 h-3.5 rounded-full bg-[#E63946]';
    }
}

function updateLastPacketInfo(packet) {
    const el = document.getElementById('last-packet');
    if (el) el.textContent = `Packet: ${packet.length > 30 ? packet.substring(0, 30) + '...' : packet}`;
}

// ========== PROCESS LIVE WEBSOCKET MESSAGES ==========
function processLiveMessage(data) {
    try {
        const delimiter = data.includes('|') ? '|' : ',';
        const parts = data.split(delimiter);
        const type = parts[0];

        if (type === 'STATS') {
            const frames = parseInt(parts[1]) || 0;
            const persons = parseInt(parts[2]) || 0;
            const objects = parseInt(parts[3]) || 0;
            const avgConf = parts[4] || '0.0%';

            document.getElementById('total-frames-count').textContent = frames;
            document.getElementById('unique-persons-count').textContent = persons;
            document.getElementById('avg-confidence').textContent = avgConf;
            
            // Dynamic chart update
            const nowTime = new Date().toLocaleTimeString();
            const chart = state.charts['activityChart'];
            if (chart) {
                if (chart.data.labels.length > 30) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                    chart.data.datasets[1].data.shift();
                }
                chart.data.labels.push(nowTime);
                chart.data.datasets[0].data.push(persons);
                chart.data.datasets[1].data.push(objects);
                chart.update('none');
            }
        } 
        else if (type === 'STATUS') {
            const status = parts[1].toUpperCase();
            const statusBadge = document.getElementById('session-status-badge');
            const statusDesc = document.getElementById('session-status-desc');
            const statusBanner = document.getElementById('status-banner');
            const bannerText = document.getElementById('banner-text');

            statusBadge.textContent = status;
            if (status === 'THEFT') {
                statusBadge.className = 'text-2xl font-black text-[#E63946] tracking-wider glow-red';
                statusDesc.textContent = 'CRITICAL: Verified theft incident confirmed!';
                statusBanner.className = 'alert-banner alert-critical';
                bannerText.textContent = 'CRITICAL ALERT - ACTIVE THEFT CONFIRMED';
            } else if (status === 'SUSPICIOUS') {
                statusBadge.className = 'text-2xl font-black text-[#F39C12] tracking-wider glow-yellow';
                statusDesc.textContent = 'WARNING: Suspicious activity detected';
                statusBanner.className = 'alert-banner alert-warning';
                bannerText.textContent = 'WARNING - SUSPICIOUS BEHAVIOUR DETECTED';
            } else {
                statusBadge.className = 'text-2xl font-black text-[#2ECC71] tracking-wider';
                statusDesc.textContent = 'No suspicious activity detected';
                statusBanner.className = 'alert-banner alert-normal';
                bannerText.textContent = 'Surveillance Active - System Monitoring Healthy';
            }
        } 
        else if (type === 'SUSPECT') {
            const suspectId = parts[1];
            const photo = parts[2];
            const timestamp = parts[3];
            const details = parts.slice(4).join(delimiter);
            
            const suspectsGrid = document.getElementById('suspects-grid');
            if (suspectsGrid.querySelector('.border-dashed')) {
                suspectsGrid.innerHTML = '';
            }
            
            if (!document.getElementById(`suspect-card-${suspectId}`)) {
                const card = document.createElement('div');
                card.id = `suspect-card-${suspectId}`;
                card.className = 'bg-[#1A1E23] p-3 rounded-lg border border-[#E63946] border-opacity-35 flex flex-col justify-between hover:border-[#E63946] transition';
                card.innerHTML = `
                    <div class="suspect-img-container mb-2 h-28 flex items-center justify-center bg-black bg-opacity-40">
                        <img src="${photo}" onerror="this.src='https://placehold.co/150x200/222/ef4444?text=Photo'" class="max-h-full max-w-full object-contain cursor-pointer rounded" onclick="openImageModal('${photo}', 'Suspect Person ${suspectId}')">
                    </div>
                    <div>
                        <h3 class="font-bold text-[#E63946] text-xs">Suspect ID ${suspectId}</h3>
                        <p class="text-[10px] text-[#A7ADB5] mt-0.5">Time: ${timestamp}</p>
                        <p class="text-[10px] text-[#F5F5F5] mt-1 bg-[#E63946] bg-opacity-10 p-1.5 rounded leading-tight border border-[#E63946] border-opacity-20 font-sans">
                            ${details}
                        </p>
                    </div>
                `;
                suspectsGrid.prepend(card);
                
                const countBadge = document.getElementById('suspects-count');
                const currentCount = suspectsGrid.querySelectorAll('.suspect-img-container').length;
                countBadge.textContent = `${currentCount} Suspect${currentCount !== 1 ? 's' : ''}`;
            }
        } 
        else if (type === 'ID_PHOTO') {
            const trackId = parts[1];
            const photo = parts[2];
            const container = document.getElementById('id-photos-container');
            
            if (container.querySelector('.border-dashed')) {
                container.innerHTML = '';
            }
            
            if (!document.getElementById(`id-photo-card-${trackId}`)) {
                const card = document.createElement('div');
                card.id = `id-photo-card-${trackId}`;
                card.className = 'id-photo-card flex flex-col items-center justify-between';
                card.innerHTML = `
                    <div class="w-full aspect-square bg-[#0A0C0D] flex items-center justify-center overflow-hidden rounded border border-[#31363F] mb-2">
                        <img src="${photo}" onerror="this.src='https://placehold.co/100/222/2ecc71?text=Photo'" class="max-h-full max-w-full object-contain rounded" onclick="openImageModal('${photo}', 'Tracked ID ${trackId}')">
                    </div>
                    <div class="text-xs font-bold text-[#A7ADB5] text-center">ID ${trackId}</div>
                `;
                container.appendChild(card);
            }
        }
        else if (type === 'FINISHED_DATA') {
            const jsonDataStr = parts.slice(1).join(delimiter);
            try {
                const data = JSON.parse(jsonDataStr);
                state.sessionData = data;
                populateSurveillanceDashboard(data);
                document.getElementById('loaded-file-name').textContent = `Live Telemetry Session Loaded`;
            } catch (err) {
                console.error('Error parsing live finished JSON data:', err);
            }
        }
        else if (type === 'FINISHED') {
            const jsonPath = parts[1];
            fetch(jsonPath)
                .then(res => res.json())
                .then(data => {
                    state.sessionData = data;
                    populateSurveillanceDashboard(data);
                    document.getElementById('loaded-file-name').textContent = `Auto-loaded: ${jsonPath}`;
                })
                .catch(err => {
                    console.error('Failed to load final session JSON:', err);
                });
        }
    } catch (err) {
        console.error('Error processing live websocket message:', err);
    }
}
