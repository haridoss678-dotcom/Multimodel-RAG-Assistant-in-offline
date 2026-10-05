// Small rendering helpers. No framework: this is a single page with a handful
// of lists, and a build step is explicitly out of scope.

export const $  = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v === null || v === undefined) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const clear = (node) => { node.replaceChildren(); return node; };

/** mm:ss for a citation timestamp - "@1923s" is not something a person can
 *  locate in a recording, "@32:03" is. */
export function clock(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
}

export function duration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  return `${m}m ${String(Math.round(seconds % 60)).padStart(2, '0')}s`;
}

/** Deliberately minimal and escaping-first: fenced code, inline code,
 *  bold, and paragraphs. The answer is model output, so it is inserted as
 *  TEXT nodes throughout - never as HTML. */
export function renderMarkdown(text) {
  const frag = document.createDocumentFragment();
  const parts = String(text ?? '').split(/```/);
  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      const body = part.replace(/^[a-zA-Z0-9_-]*\n/, '');
      frag.append(el('pre', {}, el('code', { text: body })));
      return;
    }
    for (const para of part.split(/\n{2,}/)) {
      if (!para.trim()) continue;
      frag.append(inlineParagraph(para.trim()));
    }
  });
  return frag;
}

function inlineParagraph(text) {
  const p = el('p');
  // Split on inline code and bold, keeping the delimiters' content as text.
  const tokens = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
  for (const token of tokens) {
    if (!token) continue;
    if (token.startsWith('`') && token.endsWith('`') && token.length > 2) {
      p.append(el('code', { text: token.slice(1, -1) }));
    } else if (token.startsWith('**') && token.endsWith('**') && token.length > 4) {
      p.append(el('strong', { text: token.slice(2, -2) }));
    } else {
      p.append(document.createTextNode(token));
    }
  }
  return p;
}

export function toast(message, kind = 'err') {
  const node = el('div', { class: kind === 'err' ? 'err' : 'card', role: 'status' }, message);
  Object.assign(node.style, {
    position: 'fixed', right: '18px', bottom: '18px', zIndex: 60, maxWidth: '460px',
  });
  document.body.append(node);
  setTimeout(() => node.remove(), 9000);
  return node;
}
