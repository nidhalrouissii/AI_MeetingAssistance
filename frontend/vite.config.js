import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Le proxy redirige tous les appels /api vers le backend FastAPI.
// Le frontend appelle fetch("/api/upload") sans se soucier du port,
// et aucune configuration CORS n'est nécessaire côté backend.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
