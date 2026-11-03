import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { mkdtemp, readFile, rm, rmdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';

const experiment = fileURLToPath(new URL('../experiments/ball_detection/', import.meta.url));
const maxBytes = 1024 ** 3;

export function localPlayerTracking(): Plugin {
  let busy = false;
  async function handle(req: IncomingMessage, res: ServerResponse, next: () => void) {
    if (req.url !== '/api/player-tracks') return next();
    const send = (status: number, body: object) => {
      if (res.destroyed) return;
      res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify(body));
    };
    const host = req.headers.host ?? '';
    if (!/^(localhost|127\.0\.0\.1|\[::1])(?::\d+)?$/.test(host) ||
        !['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress ?? '') ||
        req.headers.origin !== `http://${host}`) {
      return send(403, { error: 'Open PongMetrics on localhost to generate tracks.' });
    }
    if (req.method !== 'POST' || req.headers['content-type'] !== 'application/octet-stream') {
      return send(405, { error: 'Expected a video upload.' });
    }
    const length = Number(req.headers['content-length']);
    if (!Number.isSafeInteger(length) || length <= 0 || length > maxBytes) {
      return send(413, { error: 'Choose a video smaller than 1 GB.' });
    }
    if (busy) return send(409, { error: 'Another video is being analyzed. Try again shortly.' });
    busy = true;
    let folder: string | undefined;
    const abort = new AbortController();
    const cancel = () => { if (!res.writableEnded) abort.abort(); };
    res.on('close', cancel);
    try {
      folder = await mkdtemp(join(tmpdir(), 'pongmetrics-'));
      const video = join(folder, 'upload.video');
      const output = join(folder, 'tracks.json');
      await pipeline(req, createWriteStream(video), { signal: abort.signal });
      const python = process.env.PONGMETRICS_PYTHON || (process.platform === 'win32'
        ? join(experiment, '.venv', 'Scripts', 'python.exe')
        : join(experiment, '.venv', 'bin', 'python'));
      await new Promise<void>((resolve, reject) => {
        const child = spawn(python, [join(experiment, 'player_track.py'), video, '--start', '0',
          '--end', 'inf', '--no-review-sheet', '--output', output],
          { cwd: experiment, windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'], signal: abort.signal });
        let error = '';
        let spawnError: Error | undefined;
        child.stderr.on('data', chunk => { error = (error + String(chunk)).slice(-2000); });
        child.on('error', err => { spawnError = err; });
        child.on('close', code => {
          if (spawnError) reject(spawnError);
          else if (code === 0) resolve();
          else reject(new Error(error.trim() || 'Player tracking failed.'));
        });
      });
      send(200, JSON.parse(await readFile(output, 'utf8')));
    } catch (error) {
      if (!abort.signal.aborted) {
        console.error('Player tracking:', error);
        send(500, { error: 'Could not generate player tracks. Check the local Python setup and video format.' });
      }
    } finally {
      res.off('close', cancel);
      if (folder) {
        await Promise.allSettled(['upload.video', 'tracks.json'].map(name => rm(join(folder!, name), { force: true })));
        await rmdir(folder).catch(console.error);
      }
      busy = false;
    }
  }
  return {
    name: 'local-player-tracking',
    configureServer(server) { server.middlewares.use((req, res, next) => { void handle(req, res, next); }); },
    configurePreviewServer(server) { server.middlewares.use((req, res, next) => { void handle(req, res, next); }); },
  };
}
