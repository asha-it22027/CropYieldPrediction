document.addEventListener('DOMContentLoaded', () => {
    console.log('Charts.js loaded');

    const yieldChartCanvas = document.getElementById('yieldChart');
    const soilChartCanvas = document.getElementById('soilChart');

    if (!yieldChartCanvas || !soilChartCanvas) {
        console.error('Chart canvases not found!');
        return;
    }

    // Initialize default charts
    const yieldChart = new Chart(yieldChartCanvas, {
        type: 'bar',
        data: {
            labels: ['Predicted Yield'],
            datasets: [{
                label: 'Yield (tons/hectare)',
                data: [0],
                backgroundColor: '#4CAF50'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    const soilChart = new Chart(soilChartCanvas, {
        type: 'radar',
        data: {
            labels: ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall'],
            datasets: [{
                label: 'Soil Parameters',
                data: [0, 0, 0, 0, 0],
                backgroundColor: 'rgba(3, 169, 244, 0.2)',
                borderColor: '#03A9F4',
                pointBackgroundColor: '#03A9F4'
            }]
        },
        options: {
            responsive: true,
            elements: { line: { borderWidth: 3 } }
        }
    });

    // Function to update charts with new data
    window.updateCharts = function(data) { // Make global to ensure accessibility
        console.log('Updating charts with:', data);
        yieldChart.data.datasets[0].data = [data.predicted_yield];
        yieldChart.update();

        soilChart.data.datasets[0].data = [
            data.nitrogen,
            data.phosphorus,
            data.potassium,
            data.ph,
            data.rainfall
        ];
        soilChart.update();
    };

    document.getElementById('downloadChart').addEventListener('click', () => {
        console.log('Download chart clicked');
        const link = document.createElement('a');
        link.download = 'crop-prediction-charts.png';
        link.href = yieldChartCanvas.toDataURL();
        link.click();
    });
});