import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the page with hot reload and forwards /api to the Python server.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
