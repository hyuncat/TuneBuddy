// Tiny pub/sub used to drive imperative pushes (score-viewer time/annotations/
// load) from their real point of mutation, instead of a Svelte $effect
// watching state across a component boundary - see App.svelte's onMount
// subscriptions for why (a cross-component reactive read "proved unreliable
// in practice" for exactly this purpose).
//
// label (optional) is only for the console.error below - it's how a thrown
// listener gets traced back to which channel (tick vs noteDataLoaded, etc.)
// broke, instead of an anonymous "listener threw" with no context.
export function makeNotifier(label) {
  const fns = new Set();
  return {
    on: (fn) => (fns.add(fn), () => fns.delete(fn)),
    // Each listener is isolated in its own try/catch (not one try/catch
    // around the whole loop) so a listener that throws can't silently skip
    // every listener registered after it - confirmed reachable today, not
    // hypothetical: window.timeChanged throws past the end of a score (see
    // viewer.js), which pushPlaybackTime can hit on the last tick or two of
    // any full playback.
    notify: (...args) => {
      for (const fn of fns) {
        try {
          fn(...args);
        } catch (err) {
          console.error(`[notifier${label ? `:${label}` : ""}] listener threw`, err);
        }
      }
    },
  };
}
