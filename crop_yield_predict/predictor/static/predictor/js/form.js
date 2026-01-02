document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    if (!form) {
        console.error('Form not found!');
        return;
    }
    console.log('Form found, attaching listener');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log('Form submitted, fetch starting');

        const formData = new FormData(form);
        console.log('Sending FormData:', Array.from(formData.entries()));

        try {
            const url = '/';
            console.log('Fetch URL:', url);

            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            console.log('Response Status:', response.status);
            const responseText = await response.text();
            console.log('Raw Response:', responseText);

            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.status} - ${responseText}`);
            }

            const data = JSON.parse(responseText);
            console.log('Parsed Response:', data);

            if (data.error) {
                throw new Error(`Server error: ${data.error}`);
            }

            // Prepare data for charts
            const chartData = {
                nitrogen: parseFloat(formData.get('nitrogen')) || 0,
                phosphorus: parseFloat(formData.get('phosphorus')) || 0,
                potassium: parseFloat(formData.get('potassium')) || 0,
                temperature: parseFloat(formData.get('temperature')) || 0,
                humidity: parseFloat(formData.get('humidity')) || 0,
                ph: parseFloat(formData.get('ph')) || 0,
                rainfall: parseFloat(formData.get('rainfall')) || 0,
                crop: formData.get('crop') || '',
                predicted_yield: data.predicted_yield
            };
            console.log('Chart Data:', chartData); // Debug chart data

            // Update visualization section
            updateCharts(chartData); // Call chart update
            const predictionText = document.getElementById('predictionText');
            const yieldValue = document.getElementById('yieldValue');
            const downloadChart = document.getElementById('downloadChart');

            if (!predictionText || !yieldValue || !downloadChart) {
                console.error('Visualization elements not found!');
                return;
            }

            predictionText.style.display = 'block';
            yieldValue.textContent = data.predicted_yield.toFixed(2);
            downloadChart.style.display = 'inline-block';
            console.log('Visualization updated');
        } catch (error) {
            console.error('Error:', error);
            alert(`An error occurred: ${error.message}`);
        }
    });
});