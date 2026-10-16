import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `pnpm dev` serves the screens with live reload and forwards /api to the
// Python server (`python -m app.api`, port 8502). `pnpm build` writes web/dist,
// which that same Python server then serves on its own. NLT_API_PORT points the
// preview at a second copy of the server (one running a work branch's code).
const apiPort = process.env.NLT_API_PORT ?? "8502";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": `http://127.0.0.1:${apiPort}` } },
});
