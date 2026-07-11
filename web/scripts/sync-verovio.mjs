// Copies resources/verovio (the vendored Verovio WASM toolkit + viewer.js/html,
// shared with the desktop app's ScoreViewer) into web/web-resources/verovio, so
// Vite can serve it. Run on demand (`npm run sync-verovio`) whenever
// resources/verovio changes — this is NOT wired into predev/prebuild on
// purpose: it's a rarely-changing vendored asset, and auto-copying it on every
// dev/build run would add a filesystem-copy failure mode to the most
// frequently run command in the project for something that almost never needs
// refreshing. Run this explicitly instead.
import { cp, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const webDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const projectRoot = path.dirname(webDir);
const source = path.join(projectRoot, "resources", "verovio");
const dest = path.join(webDir, "web-resources", "verovio");

if (!existsSync(source)) {
  console.error(`sync-verovio: source not found at ${source}`);
  process.exit(1);
}

await rm(dest, { recursive: true, force: true });
await cp(source, dest, { recursive: true });
console.log(`sync-verovio: copied ${source} -> ${dest}`);
