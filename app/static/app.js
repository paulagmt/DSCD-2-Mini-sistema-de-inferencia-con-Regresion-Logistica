const form = document.getElementById('predict-form');
const result = document.getElementById('result');
const historyBody = document.getElementById('history-body');
const historyTable = document.getElementById('history-table');
const historyEmpty = document.getElementById('history-empty');
const refreshHistoryButton = document.getElementById('refresh-history');
const requestPreview = document.getElementById('request-preview');

const API_BASE = window.API_BASE || '';

function buildPayload() {
  const data = new FormData(form);

  return {
    age: Number(data.get('age')),
    job: data.get('job'),
    marital: data.get('marital'),
    education: data.get('education'),
    balance: Number(data.get('balance')),
    housing: data.get('housing'),
    loan: data.get('loan'),
    campaign: Number(data.get('campaign'))
  };
}

function updatePreview() {
  const payload = buildPayload();
  requestPreview.textContent = JSON.stringify(payload, null, 2);
}

async function loadHistory() {
  try {
    const response = await fetch(`${API_BASE}/api/predictions?limit=5`);
    const rows = await response.json();

    if (!response.ok) {
      throw new Error(rows.detail || 'No se pudo cargar el historial');
    }

    historyBody.innerHTML = rows.map(row => `
      <tr>
        <td>${row.created_at}</td>
        <td>${row.age}</td>
        <td>${row.job}</td>
        <td>${Number(row.balance).toFixed(2)}</td>
        <td>${(Number(row.probability) * 100).toFixed(1)}%</td>
        <td>${row.prediction === 'yes'
          ? 'Potencialmente interesado'
          : 'Baja propensión'}</td>
      </tr>
    `).join('');

    const hasRows = rows.length > 0;

    historyEmpty.classList.toggle('hidden', hasRows);
    historyTable.classList.toggle('hidden', !hasRows);

  } catch (error) {
    historyEmpty.textContent =
      `No se pudo cargar el historial: ${error.message}`;

    historyEmpty.classList.remove('hidden');
    historyTable.classList.add('hidden');
  }
}

form.addEventListener('input', updatePreview);
form.addEventListener('change', updatePreview);

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const payload = buildPayload();

  updatePreview();

  result.classList.remove('hidden');
  result.textContent = 'Ejecutando inferencia...';

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const body = await response.json();

    if (!response.ok) {
      throw new Error(body.detail || 'La solicitud fue rechazada');
    }

    result.innerHTML = `
      <strong>Probabilidad estimada:</strong>
      ${(body.probability * 100).toFixed(1)}%<br>

      <strong>Predicción:</strong>
      ${body.prediction}<br>

      <strong>Clasificación:</strong>
      ${body.classification}
    `;

    await loadHistory();

  } catch (error) {
    result.textContent = `Error: ${error.message}`;
  }
});

refreshHistoryButton.addEventListener('click', loadHistory);

updatePreview();
loadHistory();