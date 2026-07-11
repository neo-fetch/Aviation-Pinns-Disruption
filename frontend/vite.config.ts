import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies REST and the simulation WebSocket to the FastAPI
// backend (uvicorn server.app:app --port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
