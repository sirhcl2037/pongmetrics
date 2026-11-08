const $ = id => document.getElementById(id);
const config = window.CLIPS_CONFIG || {};
let apiBase = (sessionStorage.getItem('clips.apiBase') || config.apiBase || '').replace(/\/$/, '');
let password = sessionStorage.getItem('clips.password') || '';
let video = null, activeJob = null, polling = false, transferPoll = null;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const time = seconds => {
  const hours = Math.floor(seconds / 3600), minutes = Math.floor(seconds % 3600 / 60);
  const s = (seconds % 60).toFixed(1).padStart(4, '0');
  return `${hours ? `${hours}:` : ''}${String(minutes).padStart(2, '0')}:${s}`;
};
function seconds(text) {
  if (!/^\d+(?::\d{1,2}){0,2}(?:\.\d{1,3})?$/.test(text.trim())) return NaN;
  const p = text.trim().split(':').map(Number);
  return p.slice(1).some(v => v >= 60) ? NaN : p.reduce((a, b) => a * 60 + b, 0);
}
function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
async function api(route, options = {}) {
  let response;
  try { response = await fetch(`${apiBase}${route}`, { ...options, headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(password ? { Authorization: `Bearer ${password}` } : {}) }, signal: AbortSignal.timeout(20000) }); }
  catch { throw new Error('Cannot reach the video server. Check Connection settings, or make sure the local server is running.'); }
  let body;
  try { body = await response.json(); } catch { throw new Error('The video backend is not configured here. Set its URL in Connection settings.'); }
  if (!response.ok) throw new Error(body.error || `Server returned ${response.status}.`);
  return body;
}
function lock(value) {
  for (const el of document.querySelectorAll('#source-form input, #source-form button, #export-form input, #export-form select, #export-form button, #connection-form input, #connection-form button, input[type=range]')) el.disabled = value;
}
function renderRange(fromSlider = false) {
  if (!video) return;
  if (fromSlider) { $('start').value = time(Number($('start-range').value)); $('end').value = time(Number($('end-range').value)); }
  const start = seconds($('start').value), end = seconds($('end').value);
  const valid = Number.isFinite(start) && Number.isFinite(end) && end > start && end <= video.duration + .05;
  $('clip-duration').textContent = valid ? `${(end - start).toFixed(1)}s` : '—';
  $('segment-label').textContent = valid ? `${time(start)} → ${time(end)}` : 'Check the selected times';
  if (valid) {
    if (!fromSlider) { $('start-range').value = start; $('end-range').value = end; }
    $('range-fill').style.left = `${start / video.duration * 100}%`;
    $('range-fill').style.right = `${100 - end / video.duration * 100}%`;
  }
  $('preview').disabled = !valid;
  return valid ? { start, end } : null;
}
function formatNote() {
  const format = video?.formats.find(f => f.id === $('quality').value);
  const mp4 = $('container').querySelector('[value="mp4"]');
  mp4.disabled = Boolean(format?.hdr);
  if (format?.hdr) $('container').value = 'mkv';
  $('format-note').textContent = $('container').value === 'mp4'
    ? 'Precise start and end. Re-encodes to H.264 + AAC at the selected resolution and frame rate. Encoding takes time and is not lossless.'
    : 'Copies the source encoding, resolution, frame rate, and HDR. Fast, with no re-encoding. The cut can begin at an earlier keyframe. MKV usually needs VLC and may not play in PongMetrics.';
}
function showVideo(result) {
  video = result;
  $('editor').hidden = false;
  $('video-title').textContent = video.title;
  $('video-channel').textContent = `${video.channel} · ${video.formats.length} available streams`;
  $('source-duration').textContent = time(video.duration);
  $('range-duration').textContent = time(video.duration);
  $('watch-source').href = `https://www.youtube.com/watch?v=${video.id}`;
  $('quality').replaceChildren(...video.formats.map(f => {
    const option = document.createElement('option'); option.value = f.id;
    option.textContent = `${f.height}p${f.fps ? ` · ${Math.round(f.fps)} fps` : ''}${f.hdr ? ' · HDR' : ''} · ${f.codec.split('.')[0]} · ${f.extension.toUpperCase()} [${f.id}]`;
    return option;
  }));
  $('quality').value = video.formats[0].id;
  $('container').value = 'mkv';
  $('start').value = '00:00.0'; $('end').value = time(Math.min(30, video.duration));
  for (const id of ['start-range', 'end-range']) $(id).max = video.duration;
  $('limit-note').textContent = `Up to ${Math.floor(video.maxClipSeconds / 60)} minutes per clip · no resolution cap`;
  $('source-player').replaceChildren();
  const placeholder = document.createElement('p'); placeholder.className = 'hint'; placeholder.textContent = 'Choose your times, then preview the selection.'; $('source-player').append(placeholder);
  renderRange(); formatNote();
}
function showResult(result, expiresAt) {
  $('result').hidden = false; $('result-name').textContent = result.filename;
  $('result-detail').textContent = `${result.container.toUpperCase()} · streams directly to your download · link expires ${new Date(expiresAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  const file = `${apiBase}${result.fileUrl}`;
  $('download').href = `${file}&download=1`; $('download').download = result.filename;
  $('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
async function follow(job) {
  activeJob = job.id; polling = true; $('job-panel').hidden = false; lock(true);
  $('cancel').disabled = false; $('result').hidden = true;
  let failures = 0;
  try {
    while (polling) {
      $('job-stage').textContent = job.stage;
      $('job-detail').textContent = job.kind === 'inspect' ? 'Checking the video and all downloadable qualities.' : 'Processing on the server. High resolution clips can take several minutes.';
      if (job.progress > 0) $('progress').value = job.progress; else $('progress').removeAttribute('value');
      if (job.status === 'done') {
        if (job.kind === 'inspect') showVideo(job.result);
        else {
          showResult(job.result, job.expiresAt);
          $('download').onclick = () => watchTransfer(job.id);
        }
        return;
      }
      if (job.status === 'cancelled') { notice('Job cancelled.'); return; }
      if (job.status === 'error') throw new Error(job.error);
      await sleep(1200);
      try { job = await api(`/api/jobs/${activeJob}`); failures = 0; }
      catch (error) { if (++failures >= 3) { try { await api(`/api/jobs/${activeJob}`, { method: 'DELETE' }); } catch {} throw error; } }
    }
  } finally { activeJob = null; polling = false; $('job-panel').hidden = true; lock(false); renderRange(); }
}
$('source-form').addEventListener('submit', async event => {
  event.preventDefault(); notice(); $('load').disabled = true;
  try {
    const job = await api('/api/inspect', { method: 'POST', body: JSON.stringify({ url: $('url').value }) });
    video = null; $('editor').hidden = true;
    await follow(job);
  } catch (error) { notice(error.message); } finally { $('load').disabled = false; }
});
$('export-form').addEventListener('submit', async event => {
  event.preventDefault(); notice();
  const segment = renderRange(); if (!segment) return notice('Choose valid start and end times within the video.');
  if (segment.end - segment.start > video.maxClipSeconds) return notice(`Choose a clip under ${Math.floor(video.maxClipSeconds / 60)} minutes.`);
  $('export').disabled = true;
  try { await follow(await api('/api/clips', { method: 'POST', body: JSON.stringify({ videoId: video.videoId, ...segment, formatId: $('quality').value, container: $('container').value }) })); }
  catch (error) { notice(error.message); } finally { $('export').disabled = false; }
});
$('cancel').addEventListener('click', async () => {
  if (!activeJob) return; $('cancel').disabled = true;
  try { await api(`/api/jobs/${activeJob}`, { method: 'DELETE' }); } catch (error) { notice(error.message); $('cancel').disabled = false; }
});
for (const id of ['start', 'end']) $(id).addEventListener('input', () => renderRange());
$('start-range').addEventListener('input', () => { if (Number($('start-range').value) >= Number($('end-range').value)) $('start-range').value = Math.max(0, Number($('end-range').value) - .1); renderRange(true); });
$('end-range').addEventListener('input', () => { if (Number($('end-range').value) <= Number($('start-range').value)) $('end-range').value = Math.min(video.duration, Number($('start-range').value) + .1); renderRange(true); });
$('quality').addEventListener('change', formatNote); $('container').addEventListener('change', formatNote);
$('preview').addEventListener('click', () => {
  const segment = renderRange(); if (!segment) return;
  const frame = document.createElement('iframe'); frame.title = 'Preview selected YouTube segment';
  frame.src = `https://www.youtube-nocookie.com/embed/${video.id}?start=${Math.floor(segment.start)}&end=${Math.ceil(segment.end)}&autoplay=1&rel=0`;
  frame.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen'; frame.allowFullscreen = true;
  // YouTube needs a referrer to identify the embedding site.
  frame.referrerPolicy = 'strict-origin-when-cross-origin'; $('source-player').replaceChildren(frame);
});
$('api-base').value = apiBase; $('password').value = password;
$('connection-form').addEventListener('submit', async event => {
  event.preventDefault();
  const value = $('api-base').value.trim().replace(/\/$/, '');
  if (value) {
    try { const url = new URL(value); if (url.username || url.password || url.pathname !== '/' || url.search || url.hash || (url.protocol !== 'https:' && !['localhost', '127.0.0.1'].includes(url.hostname))) throw new Error(); }
    catch { return notice('Use the HTTPS origin of the video server, without a path. Localhost HTTP is allowed for local use.'); }
  }
  apiBase = value; password = $('password').value;
  sessionStorage.setItem('clips.apiBase', apiBase); sessionStorage.setItem('clips.password', password);
  notice(); await health();
});
async function health() {
  try {
    const result = await api('/api/health');
    if (!result.ok) throw new Error('Video server unavailable.');
    $('connection-status').textContent = result.passwordRequired ? 'Server online · password required' : 'Local server online';
    if (result.passwordRequired && !password) { $('settings').open = true; notice('Enter the video server password in Connection settings to begin.'); }
  } catch (error) { $('connection-status').textContent = 'Setup needed'; $('settings').open = true; notice(error.message); }
}
if (config.build) $('build').textContent = config.build;
await health();

function watchTransfer(id) {
  clearInterval(transferPoll);
  $('result-detail').textContent = 'Connecting to YouTube… Download streams straight into the finished file.';
  let ticks = 0;
  transferPoll = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${id}`);
      if (job.transfer === 'error') { notice(job.error); $('result-detail').textContent = 'Download failed. Prepare the clip again to retry.'; clearInterval(transferPoll); }
      else if (job.transfer === 'complete') { $('result-detail').textContent = 'Transfer complete. Check your browser downloads.'; clearInterval(transferPoll); }
      else if (job.transfer === 'streaming') $('result-detail').textContent = `Streaming your clip · ${Math.round(job.progress)}% processed. Your browser shows the download.`;
      if (++ticks > 2400) clearInterval(transferPoll);
    } catch { clearInterval(transferPoll); }
  }, 1500);
}
