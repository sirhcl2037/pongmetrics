import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { localPlayerTracking } from './localPlayerTracking.ts'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const repository = fileURLToPath(new URL('..', import.meta.url))
function git(...args: string[]) {
  try {
    return execFileSync('git', args, { cwd: repository, encoding: 'utf8', timeout: 3000, stdio: ['ignore', 'pipe', 'pipe'] }).trim()
  } catch { return null }
}

// Azure/CI may build from an archive with no .git directory.
const revision = process.env.PONGMETRICS_COMMIT_SHA || process.env.BUILD_SOURCEVERSION ||
  process.env.GITHUB_SHA || process.env.SCM_COMMIT_ID || git('rev-parse', 'HEAD') || ''
const commit = /^[a-f0-9]{7,40}$/i.test(revision) ? revision : 'unknown'
const status = git('status', '--porcelain')
const buildInfo = { commit, dirty: status === null ? null : status.length > 0, builtAt: new Date().toISOString() }

// https://vite.dev/config/
export default defineConfig({
  define: { __BUILD_INFO__: JSON.stringify(buildInfo) },
  plugins: [react(), localPlayerTracking()],
})
