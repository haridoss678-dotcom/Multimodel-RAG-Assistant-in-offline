import * as api from './api.js';
import { $, clear, clock, duration, el, renderMarkdown, toast } from './ui.js';

/* Speech modes keep their MEASURED costs in the label. RTF is compute-seconds
   per audio-second, so 0.3x means a 20-minute recording takes ~6 minutes. Those
   numbers are what make this a real decision rather than an opaque toggle, and
   auto-detect is both the slowest option and the only wrong one on code-mixed
   speech (§7 item 18). */
const SPEECH_MODES = [
  { id: 'translate', label: 'Mixed Tamil-English → English (fastest, ~0.3x)', translate: true,  language: ''  },
  { id: 'en',        label: 'English / Tanglish, keep wording (~1.4x)',       translate: false, language: 'en' },
  { id: 'ta',        label: 'Tamil, keep Tamil script (~4x)',                 translate: false, language: 'ta' },
  { id: 'auto',      label: 'Auto-detect (slowest, avoid for mixed speech)',  translate: false, language: ''  },
];

const state = {
  conversationId: null,
  // Several sources at once. Empty = whole corpus, which is a deliberate
  // choice here rather than the only option: one answer must never blend two
  // clients' material without the user having asked for it.
  sourceIds: [],
  sources: [],
  history: [],
  busy: false,
  rerankerWarm: false,     // first question pays the reranker's cold load
};

/* ------------------------------------------------------------------ boot */

function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved) document.documentElement.dataset.theme = saved;
  $('#themeToggle').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch { /* private mode */ }
  });
}

function initPanel() {
  const app = $('#app');
  const saved = localStorage.getItem('panel');
  if (saved) app.dataset.panel = saved;
  $('#panelToggle').addEventListener('click', () => {
    app.dataset.panel = app.dataset.panel === 'open' ? 'closed' : 'open';
    try { localStorage.setItem('panel', app.dataset.panel); } catch { /* ignore */ }
  });
  $('#railToggle').addEventListener('click', () => {
    app.dataset.rail = app.dataset.rail === 'open' ? 'closed' : 'open';
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      app.dataset.rail = 'closed';
      $('#viewer').close();
    }
  });
}


async function showEnvironment() {
  try {
    const { environment: env } = await api.getHealth();
    const bits = [
      `${env.ram_available_gb.toFixed(1)}GB free`,
      `${env.cores_physical}c/${env.cores_logical}t`,
    ];
    if (!env.ffmpeg_ok) bits.push('ffmpeg MISSING');
    if (!env.ollama_ok) bits.push('Ollama unavailable');
    const node = $('#envSummary');
    node.textContent = bits.join(' · ');
    node.title = JSON.stringify(env, null, 2);
    // Measured: below ~1GB free, answers slow sharply and a model load can fail.
    if (env.ram_available_gb < 1.0) node.style.color = 'var(--warn)';
  } catch { /* health is a nicety; never block the UI on it */ }
}

/* ---------------------------------------------------------------- scope */

function renderScope() {
  const box = clear($('#scopeSelect'));
  const ready = state.sources;

  const all = el('label', { class: 'scope-row' },
    el('input', {
      type: 'checkbox', checked: state.sourceIds.length === 0,
      onchange: (e) => { if (e.target.checked) state.sourceIds = []; renderScope(); },
    }),
    el('span', {}, 'All sources'));
  box.append(all);

  for (const s of ready) {
    const on = state.sourceIds.includes(s.source_id);
    box.append(el('label', { class: 'scope-row', title: s.filename },
      el('input', {
        type: 'checkbox', checked: on, disabled: s.state !== 'READY',
        onchange: (e) => {
          state.sourceIds = e.target.checked
            ? [...state.sourceIds, s.source_id]
            : state.sourceIds.filter((id) => id !== s.source_id);
          renderScope();
        },
      }),
      el('span', { class: 'filename' }, s.filename),
      s.state !== 'READY' ? el('span', { class: 'meta' }, ` ${s.state}`) : null));
  }
  renderScopeChip();
}

function renderScopeChip() {
  const chosen = state.sources.filter((s) => state.sourceIds.includes(s.source_id));
  const chip = $('#scopeChip');
  if (chosen.length === 0) {
    chip.textContent = 'Scoped to: all sources';
  } else if (chosen.length === 1) {
    chip.textContent = `Scoped to: ${chosen[0].filename}`;
  } else {
    chip.textContent = `Scoped to: ${chosen.length} sources`;
    chip.title = chosen.map((s) => s.filename).join(', ');
  }
  $('#scopeHint').textContent = chosen.length === 1
    ? 'Timestamp questions ("around 32 minutes") work when exactly one source is selected.'
    : chosen.length === 0
      ? 'Answers may combine evidence from every source. Tick one or more to keep them apart.'
      : `Searching ${chosen.length} sources together.`;
}

/* --------------------------------------------------------------- sources */

async function refreshSources() {
  try {
    state.sources = await api.listSources();
  } catch (e) {
    toast(`Could not list sources: ${e.message}`);
    return;
  }
  // Drop any scoped source that has since been deleted, so the chip can never
  // claim a scope that no longer exists.
  const alive = new Set(state.sources.map((s) => s.source_id));
  state.sourceIds = state.sourceIds.filter((id) => alive.has(id));
  renderScope();

  const list = clear($('#sourceList'));
  if (!state.sources.length) {
    list.append(el('p', { class: 'hint' }, 'Nothing ingested yet.'));
    return;
  }
  for (const s of state.sources) {
    const terminal = ['READY', 'FAILED', 'CANCELLED'].includes(s.state);
    const tile = el('div', { class: 'tile' },
      el('div', { class: 'row-between' },
        el('div', { class: 'filename', title: s.filename }, s.filename),
        el('span', { class: `badge ${s.state === 'READY' ? 'ok' : s.state === 'FAILED' ? 'bad' : 'warn'}` }, s.state),
      ),
      el('div', { class: 'meta' }, s.source_type),
    );
    if (!terminal) tile.append(el('div', { class: 'bar-thin' }, el('i', { 'data-job': s.source_id })));

    const actions = el('div', { class: 'row-between', style: 'margin-top:8px' });
    if (['audio', 'video', 'youtube'].includes(s.source_type)) {
      actions.append(el('button', {
        class: 'tiny', disabled: s.state !== 'READY',
        title: s.state === 'READY' ? '' : `Ready once processing finishes (now ${s.state})`,
        onclick: () => downloadTranscript(s),
      }, 'Transcript'));
    } else { actions.append(el('span')); }
    actions.append(el('button', {
      class: 'tiny', onclick: () => removeSource(s),
    }, 'Delete'));
    tile.append(actions);
    list.append(tile);
  }
  pollJobs();
}

async function downloadTranscript(source) {
  try {
    const text = await api.getTranscript(source.source_id);
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    const a = el('a', { href: url, download: `${source.filename}.txt` });
    document.body.append(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch (e) { toast(`Transcript failed: ${e.message}`); }
}

async function removeSource(source) {
  if (!confirm(`Delete "${source.filename}" and everything derived from it?`)) return;
  try {
    await api.deleteSource(source.source_id);
    await refreshSources();
  } catch (e) { toast(`Delete failed: ${e.message}`); }
}

/* Only one heavy ingestion runs at a time (§29), so several uploads sit at 0%
   with no explanation unless QUEUED is shown for what has not started. */
let jobTimer = null;
async function pollJobs() {
  clearTimeout(jobTimer);
  const pending = state.sources.filter((s) => !['READY', 'FAILED', 'CANCELLED'].includes(s.state));
  if (!pending.length) return;

  let anyRunning = false;
  for (const s of pending) {
    const bar = document.querySelector(`[data-job="${s.source_id}"]`);
    if (!bar) continue;
    const job = await api.getJob(s.source_id).catch(() => null);
    const pct = job ? Math.min((job.progress_pct || 0) * 100, 100) : 0;
    bar.style.width = `${pct}%`;
    const label = bar.closest('.tile')?.querySelector('.meta');
    if (label) {
      const started = job && (job.progress_pct || 0) > 0;
      if (started) anyRunning = true;
      label.textContent = started
        ? `${(job.stage || '').replace(/_/g, ' ').toLowerCase()} · ${pct.toFixed(0)}%`
        : anyRunning ? 'queued — one heavy job runs at a time' : 'starting…';
    }
    if (job && ['READY', 'FAILED'].includes(job.stage)) { refreshSources(); return; }
  }
  jobTimer = setTimeout(pollJobs, 2000);
}

/* ---------------------------------------------------------------- upload */

function initUpload() {
  const select = $('#speechMode');
  for (const m of SPEECH_MODES) select.append(el('option', { value: m.id }, m.label));

  const input = $('#fileInput');
  const zone = $('#dropZone');
  const btn = $('#uploadBtn');
  let chosen = null;

  const choose = (file) => {
    chosen = file;
    zone.firstElementChild.textContent = file ? file.name : 'Drop a file here, or click to choose';
    btn.disabled = !file;
  };

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.style.borderColor = 'var(--accent)'; });
  zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
  zone.addEventListener('drop', (e) => {
    e.preventDefault(); zone.style.borderColor = '';
    if (e.dataTransfer.files[0]) choose(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => choose(input.files[0]));

  const opts = () => {
    const mode = SPEECH_MODES.find((m) => m.id === select.value);
    return {
      translate: mode.translate,
      language: mode.language,
      ocrLanguage: $('#ocrLang').value,
    };
  };

  btn.addEventListener('click', async () => {
    if (!chosen) return;
    btn.disabled = true;
    try {
      await api.uploadSource(chosen, opts());
      choose(null);
      await refreshSources();
    } catch (e) {
      toast(e.code === 'FILE_TOO_LARGE' ? e.message : `Upload failed: ${e.message}`);
      btn.disabled = false;
    }
  });

  $('#ytBtn').addEventListener('click', async () => {
    const url = $('#ytUrl').value.trim();
    if (!url) return;
    try {
      await api.uploadYouTube(url, opts());
      $('#ytUrl').value = '';
      await refreshSources();
    } catch (e) { toast(`YouTube failed: ${e.message}`); }
  });
}

/* --------------------------------------------------------- conversations */

async function refreshConversations() {
  let convos = [];
  try { convos = await api.listConversations(); } catch { return; }
  const list = clear($('#convoList'));
  if (!convos.length) { list.append(el('p', { class: 'hint' }, 'No past conversations.')); return; }
  for (const c of convos) {
    // Labelled by what the conversation is ABOUT - the scoped document's
    // filename, or a heading from the opening question when nothing was
    // scoped. A timestamp is not a label: it tells you when you asked, never
    // which conversation you are looking for. The date moves to the second
    // line, where it is context rather than identity.
    const when = (c.last_active_at || '').slice(0, 16).replace('T', ' ');
    const label = c.title || when || 'Untitled conversation';
    const row = el('div', { class: 'tile convo-row' });
    row.append(el('button', {
      class: 'convo-open', title: label, onclick: () => openConversation(c.conversation_id),
    }, el('div', { class: 'convo-title' }, label),
       el('div', { class: 'meta tnum' }, `${c.message_count} messages · ${when}`)));
    row.append(el('button', {
      class: 'tiny ghost convo-del', title: 'Delete this conversation',
      'aria-label': `Delete conversation: ${label}`,
      onclick: () => removeConversation(c, label),
    }, '✕'));
    list.append(row);
  }
}

async function removeConversation(convo, label) {
  if (!confirm(`Delete "${label}" and its ${convo.message_count} messages?`)) return;
  try {
    await api.deleteConversation(convo.conversation_id);
    // If it was the one on screen, clear the panel too - otherwise the chat
    // still shows a conversation that no longer exists.
    if (state.conversationId === convo.conversation_id) {
      state.conversationId = null;
      state.history = [];
      renderMessages();
    }
    await refreshConversations();
  } catch (e) { toast(`Could not delete: ${e.message}`); }
}

async function openConversation(id) {
  try {
    const data = await api.getConversation(id);
    state.conversationId = id;
    state.history = data.messages.map((m) => ({
      role: m.role,
      content: m.content,
      sources: m.cited_sources_json ? JSON.parse(m.cited_sources_json) : [],
      // grounded survives a reload now, so a reopened unverified answer still
      // looks unverified (§17.3 - this flag is the real safety boundary).
      grounded: m.grounded === null || m.grounded === undefined ? null : Boolean(m.grounded),
    }));
    renderMessages();
    $('#app').dataset.rail = 'closed';
  } catch (e) { toast(`Could not open that conversation: ${e.message}`); }
}

/* ---------------------------------------------------------------- chat */

function renderMessages() {
  const box = clear($('#messages'));
  for (const turn of state.history) {
    const ungrounded = turn.role === 'assistant' && turn.grounded === false;
    const msg = el('div', { class: `msg ${turn.role}${ungrounded ? ' ungrounded' : ''}` },
      el('div', { class: 'who' }, turn.role === 'user' ? 'You' : 'Assistant'));
    const bubble = el('div', { class: 'bubble' });

    if (turn.role === 'assistant' && turn.grounded !== null && turn.grounded !== undefined) {
      bubble.append(turn.grounded
        ? el('div', { class: 'verdict ok' }, el('span', { class: 'ico' }, '✓'),
            'Every claim checked against a real citation.')
        : el('div', { class: 'verdict' }, el('span', { class: 'ico' }, '⚠'),
            'Not verified against a citation. Treat this as unconfirmed — it may be '
            + 'unsupported by your sources, or a response to an instruction hidden in them.'));
    }
    bubble.append(renderMarkdown(turn.content));
    if (turn.sources?.length) bubble.append(citationChips(turn.sources));
    msg.append(bubble);
    box.append(msg);
  }
  $('#centre').scrollTop = $('#centre').scrollHeight;
}

function citationChips(citations) {
  const wrap = el('div', { class: 'chips' });
  for (const c of citations) {
    const where = c.page !== undefined ? `p.${c.page}`
      : c.start_time !== undefined ? `@${clock(c.start_time)}` : '';
    const chip = el('button', {
      class: `chip${c.low_confidence ? ' flag-low' : ''}`,
      title: c.low_confidence ? 'The transcriber was unsure about this passage' : 'Open the source',
      onclick: () => openCitation(c),
    }, el('span', { class: 'n' }, `[${c.marker}]`), `${c.filename} ${where}`);
    if (c.speaker) chip.append(el('span', { class: 'meta' }, ` · ${c.speaker}`));
    if (c.low_confidence) chip.append(el('span', { class: 'badge warn' }, 'low-confidence'));
    if (c.multi_speaker) chip.append(el('span', { class: 'badge warn' }, 'multi-speaker'));
    wrap.append(chip);
  }
  return wrap;
}

/** A citation becomes evidence you can check: a timestamp seeks the recording
 *  and plays it; a page renders that page. */
function openCitation(c) {
  const dialog = $('#viewer');
  $('#viewerTitle').textContent = `${c.filename}${c.page !== undefined ? ` — page ${c.page}` : ''}`;
  const body = clear($('#viewerBody'));

  if (c.start_time !== undefined) {
    const media = el('audio', { controls: true, preload: 'metadata', style: 'width:100%',
                                src: `/sources/${c.source_id}/media` });
    // Range support is what makes this work; without 206 the element cannot seek.
    media.addEventListener('loadedmetadata', () => {
      media.currentTime = c.start_time;
      media.play().catch(() => { /* autoplay may be blocked; the control is there */ });
    }, { once: true });
    body.append(
      el('p', { class: 'meta' }, `Seeking to ${clock(c.start_time)}`
        + (c.speaker ? ` · ${c.speaker}` : '')),
      media,
    );
    if (c.multi_speaker) {
      body.append(el('p', { class: 'hint' },
        'This segment contains more than one speaker, so the label is the dominant '
        + 'voice rather than the only one.'));
    }
  } else if (c.page !== undefined) {
    body.append(el('img', { src: `/sources/${c.source_id}/page/${c.page}.png`,
                            alt: `Page ${c.page} of ${c.filename}` }));
  } else {
    body.append(el('p', { class: 'hint' }, 'No previewable location for this citation.'));
  }
  dialog.showModal();
}

/* -------------------------------------------------------------- waiting */

function startWaiting(requestId) {
  const box = clear($('#waiting'));
  const bar = el('i');
  const label = el('span', {}, 'Starting…');
  const time = el('span', { class: 'meta tnum' }, '0s');
  box.append(el('div', { class: 'progress' },
    el('div', { style: 'min-width:190px' }, label),
    el('div', { class: 'bar' }, bar), time));
  if (!state.rerankerWarm) {
    box.append(el('p', { class: 'hint' },
      'First question after a server start also loads the reranker — measured at '
      + '15-23s on top of the answer itself.'));
  }
  box.append(el('div', { class: 'skeleton', style: 'width:72%' }),
             el('div', { class: 'skeleton', style: 'width:54%' }));

  const started = Date.now();
  const tick = async () => {
    const elapsed = (Date.now() - started) / 1000;
    time.textContent = duration(elapsed);
    const p = await api.getProgress(requestId);
    if (p) {
      label.textContent = `${p.label} · step ${p.index} of ${p.total}`;
      bar.style.width = `${(p.index / p.total) * 100}%`;
    }
  };
  tick();
  return setInterval(tick, 1000);
}

async function send() {
  const input = $('#question');
  const text = input.value.trim();
  if (!text || state.busy) return;

  state.busy = true;
  $('#sendBtn').disabled = true;
  input.value = '';
  state.history.push({ role: 'user', content: text });
  renderMessages();

  const requestId = crypto.randomUUID();
  const timer = startWaiting(requestId);
  try {
    const data = await api.ask({
      message: text,
      conversation_id: state.conversationId,
      source_ids: state.sourceIds,   // the fix that makes temporal retrieval reachable
      request_id: requestId,
    });
    state.conversationId = data.conversation_id;
    state.rerankerWarm = true;
    state.history.push({
      role: 'assistant', content: data.answer,
      sources: data.sources, grounded: data.grounded,
    });
    renderDiagnostics(data);
    refreshConversations();
  } catch (e) {
    state.history.push({
      role: 'assistant',
      content: e.name === 'AbortError'
        ? 'I stopped waiting for the answer. The server may still be working — check its '
          + 'terminal. If this keeps happening, close other applications: answers slow '
          + 'sharply when this machine drops below ~1GB free.'
        : `Something went wrong: ${e.message}`,
      grounded: null, sources: [],
    });
    if (e.traceId) {
      toast(el('span', {}, `Trace ${e.traceId} `,
        el('button', { class: 'tiny', onclick: () => navigator.clipboard?.writeText(e.traceId) }, 'copy')));
    }
  } finally {
    clearInterval(timer);
    clear($('#waiting'));
    state.busy = false;
    $('#sendBtn').disabled = false;
    renderMessages();
  }
}

/* ---------------------------------------------------------- diagnostics */

function renderDiagnostics(data) {
  const d = data.diagnostics || {};
  const ev = clear($('#evidenceList'));
  if (!data.sources?.length) {
    ev.append(el('p', { class: 'hint' }, 'This answer cited nothing.'));
  }
  for (const c of data.sources || []) {
    const where = c.page !== undefined ? `page ${c.page}`
      : c.start_time !== undefined ? clock(c.start_time) : '';
    ev.append(el('div', { class: 'evidence-item' },
      el('div', {}, el('span', { class: 'n' }, `[${c.marker}] `), c.filename),
      el('div', { class: 'meta' }, [where, c.speaker].filter(Boolean).join(' · ')),
      c.low_confidence ? el('span', { class: 'badge warn' }, 'low-confidence') : null));
  }

  const cc = d.citation_check || {};
  const rows = [
    ['Retrieval strategy', d.retrieval_strategy ?? '—'],
    ['Intent', d.intent ?? '—'],
    ['Answer depth', d.response_depth ?? '—'],
    ['Candidates → reranked', `${d.candidate_count ?? '—'} → ${d.reranked_count ?? '—'}`],
    ['Evidence items', `${d.evidence_items ?? 0} (${d.evidence_chars ?? 0} chars)`],
    ['Summarization fired', d.summarized ? 'yes' : 'no'],
    ['Citation markers found', cc.markers_found ? 'yes' : 'no'],
    ['Markers out of range', cc.out_of_range ? 'yes' : 'no'],
    ['Markers re-pointed', String(cc.repointed ?? 0)],
    // Derived citations are code asserting the link, not the model. Shown,
    // never hidden (§7 item 50).
    ['Citations derived in code', cc.derived ? `${cc.derived}` : 'no'],
    ['Sentences not attributable', String(cc.unmatched_sentences ?? 0)],
    ['Grounding verdict', d.grounding_verdict ?? '—'],
    ['Retries used', String(d.retries_used ?? 0)],
    ['Largest prompt', `${d.max_prompt_tokens ?? '—'} / ${d.num_ctx} tokens`],
    ['Total time', d.elapsed_ms ? duration(d.elapsed_ms / 1000) : '—'],
  ];
  const dl = el('dl', { class: 'kv' });
  for (const [k, v] of rows) { dl.append(el('dt', {}, k), el('dd', {}, v)); }

  const box = clear($('#diagnostics'));
  box.append(dl);
  if (d.rewritten_query) {
    box.append(el('p', { class: 'hint' }, `Rewritten as: "${d.rewritten_query}"`));
  }
  if (d.llm_calls?.length) {
    box.append(el('h2', { style: 'margin-top:14px' }, 'LLM calls'));
    const calls = el('dl', { class: 'kv' });
    for (const c of d.llm_calls) {
      calls.append(el('dt', {}, c.node), el('dd', {}, `${c.prompt_tokens ?? '?'} in / ${c.eval_tokens ?? '?'} out`));
    }
    box.append(calls);
  }
  if (data.trace_id) {
    box.append(el('p', { class: 'hint' }, `trace ${data.trace_id}`));
  }
}

/* ----------------------------------------------------------------- init */

function initComposer() {
  const input = $('#question');
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 190)}px`;
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $('#sendBtn').addEventListener('click', send);
  $('#newConvo').addEventListener('click', () => {
    state.conversationId = null; state.history = []; renderMessages();
    clear($('#evidenceList')).append(el('p', { class: 'hint' }, 'Ask something to see the evidence.'));
  });
  $('#viewerClose').addEventListener('click', () => $('#viewer').close());
}

initTheme();
initPanel();
initUpload();
initComposer();
showEnvironment();
refreshSources();
refreshConversations();
