import { analyzeImage, ApiError, USE_MOCK } from './api.js';

const $ = id => document.getElementById(id);
const MAX_BYTES = 10 * 1024 * 1024;
const TYPES = ['image/jpeg', 'image/png'];
const EXT = /\.(jpe?g|png)$/i;

let file = null, previewUrl = null, busy = false;

document.addEventListener('DOMContentLoaded', () => {
  const detector = $('detector'), drop = $('dropzone'), input = $('file'), errorEl = $('error');
  $('demo').hidden = !USE_MOCK;

  const setState = s => { detector.dataset.state = s; };
  const showError = msg => { errorEl.textContent = msg; errorEl.hidden = !msg; };
  const fmtSize = b => b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1024)) + ' KB';

  function validate(f) {
    if (!TYPES.includes(f.type) || !EXT.test(f.name)) return 'Unsupported file type. Please upload a JPG, JPEG or PNG image.';
    if (f.size > MAX_BYTES) return `File is too large (${fmtSize(f.size)}). The maximum size is 10 MB.`;
    if (f.size === 0) return 'This file is empty.';
    return '';
  }

  function reset() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    file = previewUrl = null; input.value = '';
    $('img').removeAttribute('src');
    $('fill').style.width = '0';
    showError('');
    setState('idle');
  }

  function choose(f) {
    if (!f || busy) return;
    const err = validate(f);
    if (err) { showError(err); return; }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    file = f; previewUrl = URL.createObjectURL(f);
    $('img').src = previewUrl;
    $('fname').textContent = f.name; $('fname').title = f.name;
    $('fsize').textContent = fmtSize(f.size);
    showError('');
    setState('ready');
  }

  async function analyze() {
    if (!file || busy) return;
    busy = true; $('analyze').disabled = true; showError('');
    setState('loading');
    try {
      const r = await analyzeImage(file);
      showResult(r);
    } catch (e) {
      showError(e instanceof ApiError ? e.message : 'Something went wrong. Please try again.');
      setState('ready');
    } finally { busy = false; $('analyze').disabled = false; }
  }

  function showResult({ kind, confidence, mock }) {
    const res = $('result'), ai = kind === 'ai';
    res.dataset.kind = kind;
    $('res-label').textContent = ai ? 'AI-Generated' : 'Real Image';
    $('res-icon').firstElementChild.setAttribute('href', ai ? '#i-bot' : '#i-check');
    $('res-conf').textContent = confidence.toFixed(1) + '%' + (mock ? ' (demo value)' : '');
    $('bar').setAttribute('aria-valuenow', Math.round(confidence));
    $('fill').style.width = '0';
    setState('result');
    requestAnimationFrame(() => requestAnimationFrame(() => { $('fill').style.width = confidence + '%'; }));
  }

  // upload interactions
  $('browse').addEventListener('click', e => { e.stopPropagation(); input.click(); });
  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', () => choose(input.files[0]));
  ['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 1) showError('Please upload one image at a time.');
    else choose(files[0]);
  });
  // stop the browser opening files dropped outside the zone
  ['dragover', 'drop'].forEach(t => window.addEventListener(t, e => e.preventDefault()));

  $('remove').addEventListener('click', () => { if (!busy) reset(); });
  $('analyze').addEventListener('click', analyze);
  $('again').addEventListener('click', reset);

  // navigation
  const burger = $('burger'), links = $('links');
  burger.addEventListener('click', () => {
    const open = links.classList.toggle('open');
    burger.setAttribute('aria-expanded', open);
  });
  links.addEventListener('click', e => {
    if (e.target.tagName === 'A') { links.classList.remove('open'); burger.setAttribute('aria-expanded', 'false'); }
  });

  const navLinks = [...links.querySelectorAll('a')];
  const io = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (en.isIntersecting) navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + en.target.id));
    });
  }, { rootMargin: '-40% 0px -55% 0px' });
  ['home', 'how', 'about', 'faq'].forEach(id => io.observe($(id)));
});
