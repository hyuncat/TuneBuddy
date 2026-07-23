// Calls POST /realign to re-run pitch-mistake alignment at a new tolerance,
// reusing the real algorithms/MistakeDetector.py rather than a JS port.
// The DP pairing uses weighted pitch/onset/duration costs; pitch tolerance then
// classifies its diagonal operations as correct or substituted. Calling Python
// keeps those pairs identical to the desktop app. Timing reclassification stays
// client-side because it is a fixed-pairs threshold check.
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
