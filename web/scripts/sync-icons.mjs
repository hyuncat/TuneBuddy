// Copies resources/icons (the SVG icon set MistakeWidget/ToleranceWidget/
// SettingsWidget use, shared with the desktop app) into
// web/web-resources/icons, so Vite can serve them. Same rationale and
// tradeoffs as sync-verovio.mjs - see that file for why this isn't wired
// into predev/prebuild automatically.
import { cp, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const webDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const projectRoot = path.dirname(webDir);
const source = path.join(projectRoot, "resources", "icons");
const dest = path.join(webDir, "web-resources", "icons");

if (!existsSync(source)) {
  console.error(`sync-icons: source not found at ${source}`);
  process.exit(1);
}

await rm(dest, { recursive: true, force: true });
await cp(source, dest, { recursive: true });
console.log(`sync-icons: copied ${source} -> ${dest}`);
