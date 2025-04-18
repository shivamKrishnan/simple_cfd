// Function to toggle UI elements based on simulation type
function toggleSimulationMode() {
    const stlInput = document.getElementById('stlInput');
    const objectPropertiesSection = document.getElementById('objectPropertiesSection');
    const domainZInput = document.getElementById('domainZInput');
    const nzInput = document.getElementById('nzInput');
    const simulationType = document.getElementById('simulationType').value;
    const submitButton = document.querySelector('.simulation-form button[type="submit"]');
    const suggestSettingsButton = document.getElementById('suggestSettings');

    if (simulationType === '3D') {
        stlInput.classList.remove('hidden');
        objectPropertiesSection.classList.add('hidden');
        domainZInput.classList.remove('hidden');
        nzInput.classList.remove('hidden');
        submitButton.textContent = 'Confirm and Upload STL';
        submitButton.setAttribute('onclick', 'confirmSTLUpload()');
    } else {
        stlInput.classList.add('hidden');
        objectPropertiesSection.classList.remove('hidden');
        domainZInput.classList.add('hidden');
        nzInput.classList.add('hidden');
        submitButton.textContent = 'Run Simulation';
        submitButton.removeAttribute('onclick');
    }
}

function toggleLESOptions() {
    const useLES = document.getElementById('use_les').value;
    const lesSmagorinskyInput = document.getElementById('les_constant_input');

    function toggleLESOptions() {
        const useLES = document.getElementById('use_les').value === 'true';
        const lesSmagorinskyInput = document.getElementById('les_constant_input');

        if (useLES) {
            lesSmagorinskyInput.classList.remove('hidden');
        } else {
            lesSmagorinskyInput.classList.add('hidden');
            // Clear the value when hidden to prevent submission of empty string
            document.getElementById('smagorinsky_constant').value = '0.1';
        }
    }
}

function fillSuggestedSettings() {
    const simulationType = document.getElementById('simulationType').value;

    // Reset LES settings
    document.getElementById('use_les').value = 'false';
    toggleLESOptions(); // Hide Smagorinsky constant input

    // If you want to set Smagorinsky constant suggestion for LES
    if (simulationType === '2D') {
        document.getElementById('use_les').value = 'true';
        toggleLESOptions(); // Show Smagorinsky constant input
        document.getElementById('smagorinsky_constant').value = '0.17';
    }

    // Common settings for both 2D and 3D
    document.getElementById('domainX').value = '10.0';
    document.getElementById('domainY').value = '5.0';
    document.getElementById('shapeRadius').value = '1.0';
    document.getElementById('reynolds').value = '200.0';
    document.getElementById('dt').value = '0.005';
    document.getElementById('num_steps').value = '5000';
    document.getElementById('plot_interval').value = '200';

    if (simulationType === '2D') {
        // 2D specific settings
        document.getElementById('nx').value = '400';
        document.getElementById('ny').value = '200';
        document.getElementById('shape').value = 'CIRCLE';
    } else {
        // 3D specific settings
        document.getElementById('domainZ').value = '5.0';
        document.getElementById('nx').value = '100';
        document.getElementById('ny').value = '50';
        document.getElementById('nz').value = '50';
    }

    // Improved notification
    showNotification('Suggested settings applied!');
}

function showNotification(message) {
    // Remove any existing notification first
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        document.body.removeChild(existingNotification);
    }

    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    document.body.appendChild(notification);

    // Remove notification after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transition = 'opacity 0.5s';
        setTimeout(() => document.body.removeChild(notification), 500);
    }, 3000);
}

function fetchSummary() {
    const summaryElement = document.getElementById('summary');
    summaryElement.innerHTML = '<div class="loading">Loading summary data...</div>';

    fetch('/get_summary')
        .then(response => response.text())
        .then(data => {
            summaryElement.textContent = data;
        })
        .catch(error => {
            console.error('Error fetching summary:', error);
            summaryElement.textContent = 'Failed to load summary. Please try again later.';
        });
}

function confirmSTLUpload() {
    const stlFile = document.getElementById('stlFile');
    if (stlFile.files.length === 0) {
        alert('Please select an STL file first');
        return;
    }

    const params = generateParamsFile();
    if (confirm(`STL file will be uploaded with these parameters:\n\n${params.content}\n\nProceed with upload?`)) {
        document.querySelector('.simulation-form').submit();
    }
}

// Modify the form submission to handle summary fetching
document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('.simulation-form');

    form.addEventListener('submit', function (event) {
        // For 3D simulation with STL upload
        if (document.getElementById('simulationType').value === '3D') {
            const stlFile = document.getElementById('stlFile');
            if (stlFile.files.length === 0) {
                alert('Please select an STL file first');
                event.preventDefault();
                return;
            }
        }

        // Use Fetch API for form submission to handle response
        event.preventDefault();
        const formData = new FormData(form);

        fetch('/run_simulation', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Simulation failed');
                }
                // Trigger summary fetch after successful simulation
                fetchSummary();

                // Trigger file download
                return response.blob();
            })
            .then(blob => {
                // Create a link to download the zip file
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'simulation_results.zip';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Simulation failed. Please check the parameters and try again.');
            });
    });
});

// Initialize page
window.onload = function () {
    fetchSummary();
    toggleSimulationMode(); // Set initial state
};

// Function to generate and show parameters file content
function generateParamsFile() {
    const simulationType = document.getElementById('simulationType').value;
    const params = {
        domainX: document.getElementById('domainX').value,
        domainY: document.getElementById('domainY').value,
        domainZ: simulationType === '3D' ? document.getElementById('domainZ').value : '0',
        shapeRadius: document.getElementById('shapeRadius').value,
        nx: document.getElementById('nx').value,
        ny: document.getElementById('ny').value,
        nz: simulationType === '3D' ? document.getElementById('nz').value : '0',
        reynolds: document.getElementById('reynolds').value,
        dt: document.getElementById('dt').value,
        num_steps: document.getElementById('num_steps').value,
        plot_interval: document.getElementById('plot_interval').value,
        shape: simulationType === '3D' ? 'CUSTOM' :
            document.getElementById('shape').options[document.getElementById('shape').selectedIndex].text
    };

    let paramsContent = `${params.domainX} ${params.domainY} ${params.domainZ}\n`;
    paramsContent += `${params.shapeRadius}\n`;
    paramsContent += `${params.nx} ${params.ny} ${params.nz}\n`;
    paramsContent += `${params.reynolds}\n`;
    paramsContent += `${params.dt}\n`;
    paramsContent += `${params.num_steps}\n`;
    paramsContent += `${params.plot_interval}\n`;
    paramsContent += `${params.shape}\n`;

    return {
        filename: `input_params_${simulationType.toLowerCase()}.txt`,
        content: paramsContent
    };
}