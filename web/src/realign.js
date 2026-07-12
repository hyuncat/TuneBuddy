// Calls POST /realign to re-run pitch-mistake alignment at a new tolerance,
// reusing the real algorithms/MistakeDetector.py rather than a JS port.
// Pitch tolerance is baked into the alignment's own DP cost matrix (not just
// a post-hoc filter on fixed pairs), so a tolerance change can genuinely
// change which notes pair with which - a hand-ported version risks silently
// diverging from what the desktop app would produce. Timing-mistake
// reclassification doesn't need this: it's a simple fixed-pairs threshold
// check, so that stays purely client-side against /analyze's existing pairs.
export async function realign(userNotes, scoreNotes, pitchTolerance, apiBaseUrl) {
  const response = await fetch(`${apiBaseUrl}/realign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_notes: userNotes,
      score_notes: scoreNotes,
      pitch_tolerance: pitchTolerance,
    }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${response.status})`);
  }

  return response.json();
}

// Debounces calls to `fn` so only the last invocation within `delayMs` of
// silence actually runs. Built for the pitch-tolerance slider, which would
// otherwise fire a network request on every pixel of drag.
export function debounce(fn, delayMs) {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delayMs);
  };
}
