let userApiKey = '';

// Handle API key input (store in JS memory only)
document.getElementById('api-key').addEventListener('input', function (e) {
    userApiKey = e.target.value;
});

// Handle form submission
const form = document.getElementById('question-form');
form.addEventListener('submit', async function (e) {
    e.preventDefault();
    const question = document.getElementById('question').value.trim();
    const llmModel = document.querySelector('input[name="llm-model"]:checked').value;
    const embeddingModel = document.querySelector('input[name="embedding-model"]:checked').value;

    // Allow submitting even if API key is empty; backend will fall back to env variable if set.
    // Only show error if backend returns 400 with missing key message.
    if (!question) {
        showToast('Please enter a question.', 'danger');
        return;
    }

    setLoading(true);
    try {
        const response = await fetch('/question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                api_key: userApiKey,
                llm_model: llmModel,
                embedding_model: embeddingModel
            })
        });
        const data = await response.json();
        if (response.ok) {
            showResult(data.response, data.cost_usd, data.build_status);
        } else if (response.status === 400 && data.error && data.error.startsWith('Missing OpenAI API key')) {
            // Show error only if backend says key is missing
            showToast(data.error, 'danger');
        } else {
            showResult('Error: ' + (data.error || 'Unknown error'), null, null);
        }
    } catch (err) {
        showResult('Network error. Please try again.', null, null);
    }
    setLoading(false);
});

// Handle Reset Cache button
const resetBtn = document.getElementById('reset-cache-btn');
resetBtn.addEventListener('click', async function () {
    setLoading(true);
    try {
        const response = await fetch('/reset-cache', { method: 'POST' });
        if (response.ok) {
            showToast('Cache cleared!', 'success');
        } else {
            showToast('Failed to clear cache.', 'danger');
        }
    } catch (err) {
        showToast('Network error.', 'danger');
    }
    setLoading(false);
});

// Show Graph button logic
const showGraphBtn = document.getElementById('show-graph-btn');
const graphContainer = document.getElementById('graph-container');
const graphImage = document.getElementById('graphImage');
showGraphBtn.addEventListener('click', async function () {
    setLoading(true);
    graphContainer.style.display = 'none';
    try {
        const response = await fetch('/show-graph');
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            graphImage.src = url;
            graphContainer.style.display = 'block';
        } else {
            const data = await response.json();
            showToast(data.error || 'Failed to load graph.', 'danger');
        }
    } catch (err) {
        showToast('Network error while loading graph.', 'danger');
    }
    setLoading(false);
});

// Show result in the UI
function showResult(answer, cost, buildStatus) {
    const resultContainer = document.getElementById('result-container');
    const answerText = document.getElementById('answer-text');
    const costInfo = document.getElementById('cost-info');
    answerText.textContent = answer || '';
    if (cost !== null && cost !== undefined) {
        costInfo.textContent = `Estimated cost: $${cost} USD`;
        costInfo.classList.remove('d-none');
    } else {
        costInfo.textContent = '';
        costInfo.classList.add('d-none');
    }
    // Build status
    const statusDiv = document.getElementById('build-status');
    if (buildStatus) {
        statusDiv.innerHTML = `
            <strong>📦 Cache Usage:</strong><br>
            • Graph: ${buildStatus.graph === "cached" ? "✅ Used from cache" : "❌ Rebuilt"}<br>
            • Triplets: ${buildStatus.triplets === "cached" ? "✅ Used from cache" : "❌ Rebuilt"}<br>
            • Index: ${buildStatus.index === "cached" ? "✅ Used from cache" : "❌ Rebuilt"}
        `;
        statusDiv.style.display = "block";
    } else {
        statusDiv.style.display = "none";
    }
    resultContainer.classList.remove('d-none');
}

// Show toast notification
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-bg-${type} border-0 show mb-2`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}

// Loading state (disable form/buttons)
function setLoading(isLoading) {
    form.querySelectorAll('input, button').forEach(el => el.disabled = isLoading);
    resetBtn.disabled = isLoading;
    const submitBtn = document.getElementById('submit-btn');
    const submitBtnText = document.getElementById('submit-btn-text');
    const submitBtnSpinner = document.getElementById('submit-btn-spinner');
    if (isLoading) {
        submitBtn.disabled = true;
        submitBtnText.textContent = 'Loading...';
        submitBtnSpinner.classList.remove('d-none');
    } else {
        submitBtn.disabled = false;
        submitBtnText.textContent = 'Submit';
        submitBtnSpinner.classList.add('d-none');
    }
}
