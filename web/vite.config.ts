import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `pnpm dev` serves the screens with live reload and forwards /api to the
// Python server (`python -m app.api`, port 8502). `pnpm build` writes web/dist,
// which that same Python server then serves on its own.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8502" } },
});
