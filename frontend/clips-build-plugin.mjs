import { execFileSync } from 'node:child_process';
export default function clipsBuild() { return { name: 'clips-build', generateBundle() {
 let sha = process.env.PONGMETRICS_COMMIT_SHA || process.env.BUILD_SOURCEVERSION || process.env.GITHUB_SHA || process.env.SCM_COMMIT_ID || '';
 let dirty = false; try { if (!sha) sha = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim(); dirty = !!execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim(); } catch {}
 this.emitFile({ type: 'asset', fileName: 'clips/config.js', source: 'window.CLIPS_CONFIG = ' + JSON.stringify({ apiBase: "http://127.0.0.1:8787", build: (sha.slice(0, 8) || 'unknown') + (dirty ? '-local' : '') + ' · ' + new Date().toISOString() }) + ';' });
} }; }
