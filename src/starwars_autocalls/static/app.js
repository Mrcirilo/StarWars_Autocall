const form = document.querySelector('#prediction-form');
const errorBox = document.querySelector('#form-error');
const submitButton = document.querySelector('#submit-button');

function addOptions(element, values, formatter = value => value) {
  element.replaceChildren(...values.map(value => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = formatter(value);
    return option;
  }));
}

function renderUnderlyings(values) {
  const container = document.querySelector('#underlyings');
  container.replaceChildren(...values.map(value => {
    const label = document.createElement('label');
    label.className = 'ticker';
    const input = document.createElement('input');
    input.type = 'checkbox'; input.name = 'underlyings'; input.value = value;
    const text = document.createElement('span'); text.textContent = value;
    label.append(input, text);
    return label;
  }));
}

async function initialiseForm() {
  const response = await fetch('/metadata');
  if (!response.ok) throw new Error('No se pudieron cargar los catálogos del modelo.');
  const metadata = await response.json();
  addOptions(document.querySelector('#product_type'), metadata.product_types);
  addOptions(document.querySelector('#counterparty'), metadata.counterparties);
  addOptions(document.querySelector('#trader_id'), metadata.trader_ids);
  addOptions(document.querySelector('#observation_frequency'), metadata.observation_frequencies);
  renderUnderlyings(metadata.underlyings);
  document.querySelector('#requested_date').value = metadata.latest_market_date;
  document.querySelector('#market-coverage').textContent = `Mercado disponible hasta ${metadata.latest_market_date}.`;
}

function formPayload() {
  const values = new FormData(form);
  const number = name => Number(values.get(name));
  return {
    product_type: values.get('product_type'),
    basket_type: values.get('basket_type'),
    underlyings: values.getAll('underlyings'),
    autocall_barrier_pct: number('autocall_barrier_pct'),
    protection_barrier_pct: number('protection_barrier_pct'),
    no_call_period_months: number('no_call_period_months'),
    observation_frequency: values.get('observation_frequency'),
    quoted_implied_vol: number('quoted_implied_vol'),
    notional_credits: number('notional_credits'),
    counterparty: values.get('counterparty'),
    trader_id: values.get('trader_id'),
    requested_date: values.get('requested_date'),
    nominal_maturity_months: number('nominal_maturity_months'),
  };
}

function showResult(result) {
  const band = result.predicted_p10_months === null || result.predicted_p90_months === null
    ? 'no disponible'
    : `${result.predicted_p10_months.toFixed(1)} – ${result.predicted_p90_months.toFixed(1)} meses`;
  document.querySelector('#prediction-value').textContent = result.predicted_avg_duration_months.toFixed(1);
  document.querySelector('#prediction-band').textContent = band;
  document.querySelector('#prediction-ratio').textContent =
    `${(result.predicted_duration_ratio * 100).toFixed(0)}% de ${result.nominal_maturity_months} meses`;
  document.querySelector('#feature-count').textContent = result.feature_count;
  document.querySelector('#market-lag').textContent = `${result.market_lag_days_max} días`;
  document.querySelector('#model-name').textContent = result.model_name;
  document.querySelector('#prediction-warnings').textContent = (result.warnings || []).join(' ');
  document.querySelector('#result').classList.remove('hidden');
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  errorBox.textContent = '';
  const payload = formPayload();
  if (!payload.underlyings.length) {
    errorBox.textContent = 'Selecciona al menos un subyacente.';
    return;
  }
  submitButton.disabled = true;
  submitButton.textContent = 'CALCULANDO…';
  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Datos inválidos.');
    showResult(body);
  } catch (error) {
    errorBox.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = 'CALCULAR ESTIMACIÓN <span>→</span>';
  }
});

initialiseForm().catch(error => { errorBox.textContent = error.message; });
