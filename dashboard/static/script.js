// Configure Audio Graph
const audioCtx = document.getElementById('audioGraph').getContext('2d');
const audioGraph = new Chart(audioCtx, {
    type: 'line',
    data: { datasets: [] },
    options: {
        responsive: true,
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { type: 'linear', title: { display: true, text: 'Time (Samples)' } },
            y: { min: -1, max: 1, title: { display: true, text: 'Amplitude' } }
        }
    }
});

// Configure Simplified Graph
const simplifiedCtx = document.getElementById('simplifiedGraph').getContext('2d');
const simplifiedGraph = new Chart(simplifiedCtx, {
    type: 'line',
    data: {
        datasets: [{
            label: 'Heartbeat Pattern',
            data: [],
            borderColor: '#ff5733',
            borderWidth: 2
        }]
    },
    options: {
        responsive: true,
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { type: 'linear' },
            y: { min: -1, max: 1 }
        }
    }
});

// Configure Speedometer
const speedometerCtx = document.getElementById('speedometer').getContext('2d');
const speedometer = new Chart(speedometerCtx, {
    type: 'doughnut',
    data: {
        labels: ['Confidence', 'Remaining'],
        datasets: [{
            data: [0, 1],
            backgroundColor: ['#4caf50', '#ddd']
        }]
    },
    options: {
        responsive: true,
        rotation: -90,
        circumference: 180,
        plugins: { legend: { display: false } }
    }
});

// Update Graphs
function updateAudioGraph(waveforms) {
    waveforms.forEach((waveform, index) => {
        const formattedData = waveform.map((value, i) => ({ x: i, y: value }));
        audioGraph.data.datasets.push({
            data: formattedData,
            borderColor: `hsl(${Math.random() * 360}, 70%, 50%)`,
            borderWidth: 1
        });

        if (audioGraph.data.datasets.length > 10) {
            audioGraph.data.datasets.shift();
        }
    });
    audioGraph.update();
}

function updateSimplifiedGraph(data) {
    const simplifiedData = data.map((y, x) => ({ x, y }));
    simplifiedGraph.data.datasets[0].data = simplifiedData;
    simplifiedGraph.update();
}

function updateSpeedometer(confidence) {
    speedometer.data.datasets[0].data = [confidence / 100, 1 - confidence / 100];
    speedometer.update();
}

// Fetch Data Periodically
async function fetchData() {
    try {
        const res = await fetch('/data');
        const data = await res.json();

        if (data.waveforms) updateAudioGraph(data.waveforms);
        if (data.simplified_waveform) updateSimplifiedGraph(data.simplified_waveform);

        document.getElementById('logs').innerHTML = data.logs.map(log => `<p>${log}</p>`).join('');
        document.getElementById('predictions').innerHTML = data.predictions.map((p, i) =>
            `<p>Prediction ${i + 1}: ${p.disease}, Confidence Level: ${p.confidence.toFixed(2)}%</p>`).join('');
        document.getElementById('most-confident').innerHTML =
            `<strong>Most Confident Prediction:</strong> ${data.most_confident_prediction.disease}, Confidence: ${data.most_confident_prediction.confidence.toFixed(2)}%`;

        updateSpeedometer(data.most_confident_prediction.confidence || 0);
    } catch (err) {
        console.error(err);
    }
}

setInterval(fetchData, 1000);
