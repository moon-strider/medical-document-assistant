import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiOrigin = 'http://127.0.0.1:8080'
const devOrigin = 'http://127.0.0.1:5173'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: apiOrigin,
        changeOrigin: true,
        configure(proxy) {
          proxy.on('proxyReq', (proxyReq, req) => {
            if (req.headers.origin === devOrigin && req.headers.host === '127.0.0.1:5173') {
              proxyReq.setHeader('Origin', apiOrigin)
            }
          })
        },
      },
    },
  },
  build: { outDir: 'dist' },
})
