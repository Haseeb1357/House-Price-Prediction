/* scripts.js */
document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('model-select');
    const compareButton = document.getElementById('compare-button');
    const bestModelsButton = document.getElementById('best-models-button');
    const chartsContainer = document.getElementById('charts');
    const modelInfoContainer = document.getElementById('model-info');
    const resultsContainer = document.getElementById('results');

    modelSelect.addEventListener('change', () => {
        const modelName = modelSelect.value;
        if (modelName) {
            fetchModelResults(modelName);
        }
    });

    compareButton.addEventListener('click', () => {
        const model1 = prompt("Enter the name of the first model to compare (e.g., 'linear_regression'):");
        const model2 = prompt("Enter the name of the second model to compare:");
        if (model1 && model2) {
            fetchCompareResults(model1, model2);
        }
    });

    bestModelsButton.addEventListener('click', () => {
        fetchBestModels();
    });

    function fetchModelResults(modelName) {
        fetch(`/model/${modelName}`)
            .then(response => response.json())
            .then(data => {
                displayModelResults(data);
            })
            .catch(error => console.error('Error fetching model results:', error));
    }

    function fetchCompareResults(model1, model2) {
        fetch(`/compare/${model1}/${model2}`)
            .then(response => response.json())
            .then(data => {
                displayComparisonResults(data);
            })
            .catch(error => console.error('Error comparing models:', error));
    }

    function fetchBestModels() {
        fetch('/best_regression_classification')
            .then(response => response.json())
            .then(data => {
                displayBestModels(data);
            })
            .catch(error => console.error('Error fetching best models:', error));
    }

    function displayModelResults(data) {
        chartsContainer.style.display = 'block';
        modelInfoContainer.innerHTML = `
            <h3>Model: ${data.model_name}</h3>
            <p>${data.mse ? 'MSE: ' + data.mse : 'Accuracy: ' + data.accuracy}</p>
            <!-- Insert charts here -->
        `;
    }

    function displayComparisonResults(data) {
        chartsContainer.style.display = 'block';
        modelInfoContainer.innerHTML = `
            <h3>Model 1: ${data.model1.model_name}</h3>
            <p>${data.model1.mse ? 'MSE: ' + data.model1.mse : 'Accuracy: ' + data.model1.accuracy}</p>
            <h3>Model 2: ${data.model2.model_name}</h3>
            <p>${data.model2.mse ? 'MSE: ' + data.model2.mse : 'Accuracy: ' + data.model2.accuracy}</p>
            <!-- Insert comparison charts here -->
        `;
    }

    function displayBestModels(data) {
        chartsContainer.style.display = 'block';
        modelInfoContainer.innerHTML = `
            <h3>Best Regression Model:</h3>
            <p>MSE: ${data.best_regression.mse}</p>
            <h3>Best Classification Model:</h3>
            <p>Accuracy: ${data.best_classification.accuracy}</p>
            <!-- Insert charts for best models here -->
        `;
    }
});
