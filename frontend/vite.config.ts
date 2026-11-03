import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { localPlayerTracking } from './localPlayerTracking.ts'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), localPlayerTracking()],
})
