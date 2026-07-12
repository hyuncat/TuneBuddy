// Client-side cache for parsed score note data (POST /notedata's response),
// keyed by a SHA-256 hash of the score file's *content*, not its filename -
// two uploads can share a name (e.g. a re-exported edit of the same file),
// and a name-only key would silently serve stale data for the second one.
//
// This replaces a server-side GET-by-id endpoint entirely: the client holds
// the actual data once fetched, so there's nothing to look up by reference.
// Session-scoped only (a plain in-memory Map) - a page reload clears it, and
// the next score selection just re-fetches and re-populates it.
const cache = new Map(); // content hash -> /notedata response

async function hashFile(file) {
  const buffer = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// Returns the cached note data for `scoreFile` if this exact content has been
// seen before; otherwise POSTs it to /notedata and caches the result.
export async function getNoteData(scoreFile, apiBaseUrl) {
  const hash = await hashFile(scoreFile);
  if (cache.has(hash)) {
    return cache.get(hash);
  }

  const formData = new FormData();
  formData.append("score", scoreFile);

  const response = await fetch(`${apiBaseUrl}/notedata`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${response.status})`);
  }

  const data = await response.json();
  cache.set(hash, data);
  return data;
}
