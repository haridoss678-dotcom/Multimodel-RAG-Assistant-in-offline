// Thin wrapper over the §20 API surface. One place that knows about HTTP, so
// the rest of the UI deals in data.

/** A chat turn is measured at ~107s and can legitimately run far longer under
 *  memory pressure (§23.1). Anything under ~5 minutes fires during normal use;
 *  the client must outlast the server, not the other way round (§7 item 49). */
export const CHAT_TIMEOUT_MS = 900_000;

async function request(path, options = {}, timeoutMs = 30_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(path, { ...options, signal: controller.signal });
    if (!resp.ok) {
      let detail = null;
      try { detail = (await resp.json()).detail; } catch { /* non-JSON body */ }
      const err = new Error(detail?.message || `${resp.status} ${resp.statusText}`);
      err.status = resp.status;
      err.code = detail?.code;
      err.traceId = detail?.trace_id;
      throw err;
    }
    return resp;
  } finally {
    clearTimeout(timer);
  }
}

const json = async (path, options, timeoutMs) =>
  (await request(path, options, timeoutMs)).json();

export const getHealth        = () => json('/health');
export const listSources      = () => json('/sources');
export const getJob           = (id) => json(`/jobs/${id}`);
export const listConversations = () => json('/conversations');
export const getConversation  = (id) => json(`/conversations/${id}`);
export const getTrace         = (id) => json(`/trace/${id}`);

export const deleteConversation = (id) =>
  request(`/conversations/${id}`, { method: 'DELETE' }, 30_000);

export const deleteSource = (id) =>
  request(`/sources/${id}`, { method: 'DELETE' }, 60_000);

export const getTranscript = async (id) =>
  (await request(`/sources/${id}/transcript`, {}, 60_000)).text();

export function uploadSource(file, opts) {
  const body = new FormData();
  body.append('file', file);
  body.append('translate_to_english', String(opts.translate));
  body.append('asr_language', opts.language ?? '');
  body.append('ocr_language', opts.ocrLanguage ?? 'en');
  return json('/sources/upload', { method: 'POST', body }, 300_000);
}

export const uploadYouTube = (url, opts) =>
  json('/sources/youtube', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      url,
      translate_to_english: opts.translate,
      asr_language: opts.language ?? '',
    }),
  }, 120_000);

export const ask = (payload) =>
  json('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, CHAT_TIMEOUT_MS);

/** 404 here means "not tracked yet, or already finished" - never an error.
 *  The caller keeps waiting on the answer itself. */
export async function getProgress(requestId) {
  try {
    return await json(`/chat/progress/${requestId}`, {}, 8_000);
  } catch {
    return null;
  }
}
