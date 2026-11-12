// API service layer: the only file that talks to the backend.

export const CONFIG = {
  // Set this to your deployed prediction endpoint, e.g. "http://localhost:8000/predict".
  // Leave empty to run in MOCK mode (UI demo only).
  API_URL: window.PIXELPROOF_API_URL || '',
  FIELD_NAME: 'file',   // multipart field name your backend expects
  TIMEOUT_MS: 180000,
};

export const USE_MOCK = !CONFIG.API_URL;

export class ApiError extends Error {
  constructor(kind, message) { super(message); this.kind = kind; }
}

/** Sends the image as multipart/form-data. Resolves to { kind: 'ai'|'real', confidence: 0-100, mock: bool }. */
export async function analyzeImage(file) {
  if (USE_MOCK) return mockPredict();

  const body = new FormData();
  body.append(CONFIG.FIELD_NAME, file);

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), CONFIG.TIMEOUT_MS);
  let res;
  try {
    res = await fetch(CONFIG.API_URL, { method: 'POST', body, signal: ctrl.signal });
  } catch (e) {
    throw new ApiError('network', e.name === 'AbortError'
      ? 'The request timed out. Please try again.'
      : 'Could not reach the prediction service. Check your connection and try again.');
  } finally { clearTimeout(timer); }

  if (res.status === 400 || res.status === 415 || res.status === 422)
    throw new ApiError('invalid', 'The server could not read this image. Try a different JPG or PNG.');
  if (res.status === 413) throw new ApiError('invalid', 'This file is too large for the server.');
  if (!res.ok) throw new ApiError('server', 'The prediction service had a problem. Please try again later.');

  let data;
  try { data = await res.json(); } catch { throw new ApiError('format', 'Unexpected response from the server.'); }
  return normalize(data);
}

/** Adapt this to your backend's JSON. Accepts {prediction|label|class, confidence|probability|score}. */
function normalize(data) {
  const raw = data?.prediction ?? data?.label ?? data?.class;
  const conf = Number(data?.confidence ?? data?.probability ?? data?.score);
  let kind = null;
  if (typeof raw === 'string') {
    const s = raw.toLowerCase();
    if (/(^ai|fake|generated|synthetic)/.test(s)) kind = 'ai';
    else if (/(real|authentic|human)/.test(s)) kind = 'real';
  } else if (raw === 1) kind = 'ai';      // label AI = 1
  else if (raw === 0) kind = 'real';
  if (!kind || !Number.isFinite(conf) || conf < 0 || conf > 100)
    throw new ApiError('format', 'Unexpected response from the server.');
  return { kind, confidence: conf <= 1 ? conf * 100 : conf, mock: false };
}

// MOCK ONLY: placeholder output so the UI can be previewed. NOT a model result.
async function mockPredict() {
  await new Promise(r => setTimeout(r, 2200));
  return { kind: Math.random() < .5 ? 'ai' : 'real', confidence: 70 + Math.random() * 29, mock: true };
}
