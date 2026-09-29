import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev, API paths are proxied to FastAPI so the browser sees one origin:
// no CORS preflights, and <audio src="/songs/1/audio"> just works.
const API = process.env.VITE_API_PROXY ?? "http://localhost:8000";
const apiPaths = ["/identify", "/songs", "/health"];

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // host: true exposes the dev server on the LAN so a phone can open it.
    host: true,
    proxy: Object.fromEntries(apiPaths.map((p) => [p, { target: API, changeOrigin: true }])),
  },
});
