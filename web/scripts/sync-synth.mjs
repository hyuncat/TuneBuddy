// Copies the assets MIDI playback needs into web/web-resources/synth/:
//   - resources/MuseScore_General.sf3 (the same soundfont app.py loads into
//     pyfluidsynth - see app.py's `self.SOUNDFONT`), so the browser-side
//     synth plays with identical instrument sounds.
//   - js-synthesizer's AudioWorklet runtime (dist/js-synthesizer.worklet.js)
//     and the libfluidsynth WASM build compiled WITH libsndfile
//     (externals/libfluidsynth-2.4.6-with-libsndfile.js) - the libsndfile
//     variant specifically, since MuseScore_General.sf3 is an SF3
//     (Ogg-Vorbis-compressed) soundfont; the standard build can't load it.
// These are loaded via `AudioContext.audioWorklet.addModule(url)` at
// runtime (see playback.svelte.js), not through Vite's normal JS import
// graph, so they must be plain static files - same rationale as
// sync-verovio.mjs/sync-icons.mjs.
import { cp, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const webDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const projectRoot = path.dirname(webDir);
const dest = path.join(webDir, "web-resources", "synth");

const files = [
  {
    src: path.join(projectRoot, "resources", "MuseScore_General.sf3"),
    name: "MuseScore_General.sf3",
  },
  {
    src: path.join(
      webDir,
      "node_modules",
      "js-synthesizer",
      "externals",
      "libfluidsynth-2.4.6-with-libsndfile.js"
    ),
    name: "libfluidsynth-2.4.6-with-libsndfile.js",
  },
  {
    src: path.join(webDir, "node_modules", "js-synthesizer", "dist", "js-synthesizer.worklet.js"),
    name: "js-synthesizer.worklet.js",
  },
];

await mkdir(dest, { recursive: true });

for (const { src, name } of files) {
  if (!existsSync(src)) {
    console.error(`sync-synth: source not found at ${src}`);
    process.exit(1);
  }
  await cp(src, path.join(dest, name));
  console.log(`sync-synth: copied ${src} -> ${dest}/${name}`);
}
