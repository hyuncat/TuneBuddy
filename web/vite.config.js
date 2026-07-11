import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Respect an externally-assigned port (e.g. the dev preview harness sets
// PORT when 5173 is already taken by another project) instead of always
// binding to Vite's hardcoded default.
const port = Number(process.env.PORT) || 5173;

export default defineConfig({
  plugins: [svelte()],
  // vendored external assets (Verovio now, the MuseScore soundfont later) are
  // synced into web-resources/ via `npm run sync-verovio` rather than
  // committed twice; see web/scripts/sync-verovio.mjs.
  publicDir: "web-resources",
  server: {
    port,
    strictPort: true,
  },
});
