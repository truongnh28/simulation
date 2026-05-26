# HTTP/HTTPS/H2 + gRPC Deep Dive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write two comprehensive HTML deep-dive reference documents — `docs/http-deep-dive.html` (18 sections, HTTP/1.0→1.1→HTTPS→H2→H3) and `docs/grpc-deep-dive.html` (16 sections, gRPC + Protobuf stack bottom-up) — matching the style of the existing `docs/websocket-deep-dive.html`.

**Architecture:** Each file is a single self-contained static HTML page. White background, blue accent theme, sticky sidebar with IntersectionObserver-driven active state, inline SVG diagrams, Vietnamese language content. No build step, no external dependencies. CSS variables cloned from `websocket-deep-dive.html`.

**Tech Stack:** HTML5, CSS3 (custom properties), vanilla JS (IntersectionObserver), inline SVG.

---

## File Map

| File | Action | Description |
|------|--------|-------------|
| `docs/http-deep-dive.html` | Create | 18 sections, 7 SVG diagrams |
| `docs/grpc-deep-dive.html` | Create | 16 sections, 6 SVG diagrams |

---

## CSS Template (shared across both files)

Paste this `<style>` block verbatim into both files. It is identical to `websocket-deep-dive.html`:

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #ffffff; --bg2: #f8fafc; --bg3: #f1f5f9; --border: #e2e8f0;
  --text: #1e293b; --text2: #475569; --text3: #94a3b8;
  --blue: #0369a1; --blue-l: #e0f2fe; --blue-ll: #f0f9ff;
  --green: #15803d; --green-l: #dcfce7;
  --orange: #c2410c; --orange-l: #fff7ed;
  --red: #dc2626; --red-l: #fef2f2;
  --purple: #7c3aed; --purple-l: #faf5ff;
  --yellow: #92400e; --yellow-l: #fffbeb;
  --accent: #0ea5e9;
}
html { scroll-behavior: smooth; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.75; font-size: 15px; }

.layout { display: flex; min-height: 100vh; }
.sidebar { width: 260px; flex-shrink: 0; position: sticky; top: 0; height: 100vh; overflow-y: auto; background: var(--bg2); border-right: 1px solid var(--border); padding: 1.5rem 0; }
.sidebar-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text3); padding: 0 1.25rem 0.75rem; }
.nav-item { display: flex; align-items: center; gap: 0.6rem; padding: 0.45rem 1.25rem; font-size: 0.85rem; color: var(--text2); text-decoration: none; border-left: 3px solid transparent; transition: all 0.15s; }
.nav-item:hover, .nav-item.active { background: var(--blue-ll); color: var(--blue); border-left-color: var(--accent); }
.nav-item.active { font-weight: 600; }
.nav-num { font-size: 0.7rem; background: var(--bg3); color: var(--text3); border-radius: 4px; padding: 0.1rem 0.4rem; font-weight: 600; min-width: 1.6rem; text-align: center; }
.nav-divider { height: 1px; background: var(--border); margin: 0.5rem 1.25rem; }
.nav-group-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text3); padding: 0.6rem 1.25rem 0.2rem; }

.main { flex: 1; max-width: 820px; padding: 3rem 3rem 6rem; }
.page-title { font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem; }
.page-sub { font-size: 1rem; color: var(--text2); margin-bottom: 2rem; line-height: 1.6; }

.section { margin-bottom: 4rem; }
.section-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem; padding-bottom: 0.6rem; border-bottom: 2px solid var(--blue-l); }
.section-num { background: var(--blue); color: white; font-weight: 800; font-size: 0.75rem; border-radius: 6px; padding: 0.25rem 0.6rem; letter-spacing: 0.05em; }
.section-title { font-size: 1.35rem; font-weight: 700; color: #0f172a; }

h3 { font-size: 1rem; font-weight: 700; color: var(--blue); margin: 1.75rem 0 0.5rem; }
h4 { font-size: 0.85rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.08em; margin: 1.25rem 0 0.4rem; }
p { color: var(--text2); margin-bottom: 0.8rem; }
p strong { color: var(--text); }

pre { background: var(--bg2); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; overflow-x: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #0f4c81; line-height: 1.7; margin: 0.75rem 0 1.25rem; white-space: pre; }
code { background: var(--blue-ll); color: var(--blue); padding: 0.15em 0.45em; border-radius: 4px; font-size: 0.85em; font-family: 'JetBrains Mono', monospace; border: 1px solid #bae6fd; }

.box { border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0 1.25rem; font-size: 0.9rem; }
.box-blue   { background: var(--blue-ll);  border: 1px solid #bae6fd; color: #0c4a6e; }
.box-green  { background: var(--green-l);  border: 1px solid #86efac; color: #14532d; }
.box-orange { background: var(--orange-l); border: 1px solid #fdba74; color: #7c2d12; }
.box-yellow { background: var(--yellow-l); border: 1px solid #fde68a; color: var(--yellow); }
.box-purple { background: var(--purple-l); border: 1px solid #c4b5fd; color: #4c1d95; }
.box-red    { background: var(--red-l);    border: 1px solid #fca5a5; color: #7f1d1d; }
.box-title  { font-weight: 700; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; opacity: 0.7; }

table { width: 100%; border-collapse: collapse; margin: 0.75rem 0 1.25rem; font-size: 0.88rem; }
th { background: var(--blue-ll); color: var(--blue); padding: 0.6rem 1rem; text-align: left; font-weight: 700; border-bottom: 2px solid #bae6fd; }
td { padding: 0.55rem 1rem; border-bottom: 1px solid var(--bg3); color: var(--text2); vertical-align: top; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg2); }

ul, ol { padding-left: 1.5rem; margin-bottom: 0.8rem; }
li { margin-bottom: 0.35rem; color: var(--text2); }
li strong { color: var(--text); }

.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 0.75rem 0 1.25rem; }
.col-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.1rem; }
.col-card h4 { margin-top: 0; }

.diagram-wrap { margin: 0.75rem 0 1.25rem; overflow-x: auto; }
svg.diagram { display: block; max-width: 100%; height: auto; }

.checklist { list-style: none; padding: 0; margin: 0.75rem 0 1.25rem; }
.checklist li { display: flex; align-items: flex-start; gap: 0.65rem; padding: 0.45rem 0.75rem; border-radius: 6px; margin-bottom: 0.3rem; font-size: 0.9rem; color: var(--text2); background: var(--bg2); border: 1px solid var(--border); }
.checklist li::before { content: '☐'; font-size: 1rem; color: var(--text3); flex-shrink: 0; margin-top: 0.05rem; }
```

## IntersectionObserver Script (identical in both files)

Place before `</body>`:

```html
<script>
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    const id = e.target.id;
    const link = document.querySelector(`.nav-item[href="#${id}"]`);
    if (link) link.classList.toggle('active', e.isIntersecting);
  });
}, { rootMargin: '-10% 0px -80% 0px' });
document.querySelectorAll('.section').forEach(s => observer.observe(s));
</script>
```

## SVG Arrow Marker (include in every SVG that uses arrows)

```html
<defs>
  <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L0,6 L8,3 z" fill="#94a3b8"/>
  </marker>
  <marker id="arr-blue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L0,6 L8,3 z" fill="#0ea5e9"/>
  </marker>
  <marker id="arr-green" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L0,6 L8,3 z" fill="#22c55e"/>
  </marker>
</defs>
```

---

## Task 1 — `http-deep-dive.html`: Shell + CSS + Sidebar

**Files:** Create `docs/http-deep-dive.html`

- [ ] Create file with DOCTYPE, `<head>` (charset, viewport, title "HTTP Deep Dive"), full CSS template above
- [ ] Add sidebar HTML with 3 nav groups:

```html
<div class="layout">
<nav class="sidebar">
  <div class="sidebar-title">HTTP Deep Dive</div>
  <div class="nav-group-label">Nền tảng</div>
  <a class="nav-item" href="#s1"><span class="nav-num">01</span>HTTP là gì?</a>
  <a class="nav-item" href="#s2"><span class="nav-num">02</span>HTTP/1.0</a>
  <a class="nav-item" href="#s3"><span class="nav-num">03</span>HTTP/1.1</a>
  <a class="nav-item" href="#s4"><span class="nav-num">04</span>Methods &amp; Semantics</a>
  <a class="nav-item" href="#s5"><span class="nav-num">05</span>Headers &amp; Structure</a>
  <a class="nav-item" href="#s6"><span class="nav-num">06</span>Status Codes</a>
  <a class="nav-item" href="#s7"><span class="nav-num">07</span>URL &amp; Negotiation</a>
  <div class="nav-divider"></div>
  <div class="nav-group-label">Security &amp; Caching</div>
  <a class="nav-item" href="#s8"><span class="nav-num">08</span>Caching</a>
  <a class="nav-item" href="#s9"><span class="nav-num">09</span>Cookies &amp; Sessions</a>
  <a class="nav-item" href="#s10"><span class="nav-num">10</span>CORS</a>
  <a class="nav-item" href="#s11"><span class="nav-num">11</span>HTTPS &amp; PKI</a>
  <a class="nav-item" href="#s12"><span class="nav-num">12</span>TLS Handshake</a>
  <div class="nav-divider"></div>
  <div class="nav-group-label">HTTP/2 &amp; HTTP/3</div>
  <a class="nav-item" href="#s13"><span class="nav-num">13</span>HTTP/2 — Binary Framing</a>
  <a class="nav-item" href="#s14"><span class="nav-num">14</span>Streams &amp; Multiplexing</a>
  <a class="nav-item" href="#s15"><span class="nav-num">15</span>HPACK Compression</a>
  <a class="nav-item" href="#s16"><span class="nav-num">16</span>Flow Control &amp; Priority</a>
  <a class="nav-item" href="#s17"><span class="nav-num">17</span>Security Headers</a>
  <a class="nav-item" href="#s18"><span class="nav-num">18</span>HTTP/3 &amp; QUIC</a>
</nav>
<main class="main">
  <h1 class="page-title">HTTP Deep Dive</h1>
  <p class="page-sub">HTTP/1.0 → HTTP/1.1 → HTTPS → HTTP/2 → HTTP/3 — từng bước tiến hóa, từng bit trong frame.</p>
  <!-- SVG 1: Evolution timeline goes here -->
  <!-- sections s1–s18 go here -->
</main>
</div>
```

- [ ] Add IntersectionObserver script before `</body>`
- [ ] Open in browser — verify sidebar shows, layout is correct, no horizontal overflow

---

## Task 2 — `http-deep-dive.html`: SVG 1 — Evolution Timeline

Place immediately after `.page-sub`, before `#s1`.

- [ ] Add this SVG (viewBox="0 0 780 80"):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 780 80" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#94a3b8"/>
    </marker>
  </defs>
  <!-- Phase boxes -->
  <rect x="10" y="20" width="110" height="40" rx="6" fill="#e0f2fe" stroke="#7dd3fc" stroke-width="1.5"/>
  <text x="65" y="37" text-anchor="middle" font-size="11" font-weight="700" fill="#0369a1">HTTP/1.0</text>
  <text x="65" y="52" text-anchor="middle" font-size="9" fill="#0369a1">1996 · RFC 1945</text>

  <rect x="152" y="20" width="110" height="40" rx="6" fill="#dcfce7" stroke="#86efac" stroke-width="1.5"/>
  <text x="207" y="37" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">HTTP/1.1</text>
  <text x="207" y="52" text-anchor="middle" font-size="9" fill="#15803d">1997 · RFC 2616/7230</text>

  <rect x="294" y="20" width="110" height="40" rx="6" fill="#fef3c7" stroke="#fde68a" stroke-width="1.5"/>
  <text x="349" y="37" text-anchor="middle" font-size="11" font-weight="700" fill="#92400e">HTTPS / TLS</text>
  <text x="349" y="52" text-anchor="middle" font-size="9" fill="#92400e">2000s · RFC 2818</text>

  <rect x="436" y="20" width="110" height="40" rx="6" fill="#faf5ff" stroke="#c4b5fd" stroke-width="1.5"/>
  <text x="491" y="37" text-anchor="middle" font-size="11" font-weight="700" fill="#7c3aed">HTTP/2</text>
  <text x="491" y="52" text-anchor="middle" font-size="9" fill="#7c3aed">2015 · RFC 7540</text>

  <rect x="578" y="20" width="110" height="40" rx="6" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>
  <text x="633" y="37" text-anchor="middle" font-size="11" font-weight="700" fill="#c2410c">HTTP/3</text>
  <text x="633" y="52" text-anchor="middle" font-size="9" fill="#c2410c">2022 · RFC 9114</text>

  <!-- Arrows between boxes -->
  <line x1="121" y1="40" x2="150" y2="40" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="263" y1="40" x2="292" y2="40" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="405" y1="40" x2="434" y2="40" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="547" y1="40" x2="576" y2="40" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
</svg>
</div>
```

- [ ] Verify in browser — 5 boxes visible, arrows connecting them, no overflow

---

## Task 3 — `http-deep-dive.html`: Sections s1–s3 + SVG 2 (Connection Models)

Each section uses this HTML skeleton:
```html
<section class="section" id="sN">
  <div class="section-header">
    <span class="section-num">0N</span>
    <h2 class="section-title">Title</h2>
  </div>
  <!-- content -->
</section>
```

- [ ] **s1 — HTTP là gì?**
  - Paragraph: HTTP = HyperText Transfer Protocol, giao thức tầng ứng dụng (OSI Layer 7), chạy trên TCP/IP
  - Paragraph: stateless (mỗi request độc lập), text-based (HTTP/1.x), request-response model
  - Box blue: "RFC Overview" — RFC 1945 (HTTP/1.0), RFC 2616 → RFC 7230-7235 (HTTP/1.1), RFC 7540 (HTTP/2), RFC 9110-9114 (HTTP semantics + HTTP/3)
  - Table: Use cases — REST API, web pages, file download, streaming, webhooks, SSE

- [ ] **s2 — HTTP/1.0**
  - Paragraph: ra đời 1996, mỗi request = 1 TCP connection, sau response → đóng kết nối
  - Code block: ví dụ raw HTTP/1.0 request + response (GET / HTTP/1.0, Content-Type, Content-Length)
  - Box orange "Vấn đề": performance kém — 3-way handshake cho mỗi request, không có host header (1 IP = 1 website), no persistent connection, no chunked encoding

- [ ] **s3 — HTTP/1.1**
  - Paragraph: RFC 2616 (1997), cải tiến lớn: persistent connections (default), pipelining, chunked transfer, virtual hosts (Host header required), conditional requests
  - Sub-section "Keep-Alive": `Connection: keep-alive`, `Keep-Alive: timeout=5, max=1000`
  - Sub-section "Pipelining": gửi nhiều request không cần đợi response — nhưng phải nhận response đúng thứ tự (HOL blocking)
  - Sub-section "Chunked Transfer": `Transfer-Encoding: chunked`, format: `<hex-size>\r\n<data>\r\n`, kết thúc bằng `0\r\n\r\n`
  - Code block: chunked response example
  - **SVG 2 — Connection Models** (viewBox="0 0 720 200"):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 720 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr2" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#64748b"/>
    </marker>
  </defs>
  <!-- Column headers -->
  <text x="120" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#0369a1">HTTP/1.0</text>
  <text x="120" y="30" text-anchor="middle" font-size="9" fill="#64748b">New conn per request</text>
  <text x="360" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">HTTP/1.1 Pipeline</text>
  <text x="360" y="30" text-anchor="middle" font-size="9" fill="#64748b">Serial on 1 conn (HOL)</text>
  <text x="600" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#7c3aed">HTTP/2</text>
  <text x="600" y="30" text-anchor="middle" font-size="9" fill="#64748b">Parallel streams</text>

  <!-- HTTP/1.0: 3 separate connections -->
  <rect x="30" y="45" width="180" height="18" rx="3" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="120" y="58" text-anchor="middle" font-size="9" fill="#0369a1">TCP conn → REQ1 → RESP1 → close</text>
  <rect x="30" y="70" width="180" height="18" rx="3" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="120" y="83" text-anchor="middle" font-size="9" fill="#0369a1">TCP conn → REQ2 → RESP2 → close</text>
  <rect x="30" y="95" width="180" height="18" rx="3" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="120" y="108" text-anchor="middle" font-size="9" fill="#0369a1">TCP conn → REQ3 → RESP3 → close</text>

  <!-- HTTP/1.1: 1 connection, serial -->
  <rect x="270" y="45" width="180" height="100" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="360" y="60" text-anchor="middle" font-size="9" fill="#15803d">1 TCP connection</text>
  <rect x="280" y="67" width="50" height="14" rx="2" fill="#bbf7d0"/>
  <text x="305" y="78" text-anchor="middle" font-size="8" fill="#15803d">REQ1</text>
  <rect x="280" y="84" width="50" height="14" rx="2" fill="#86efac"/>
  <text x="305" y="95" text-anchor="middle" font-size="8" fill="#15803d">RESP1</text>
  <rect x="340" y="84" width="50" height="14" rx="2" fill="#bbf7d0"/>
  <text x="365" y="95" text-anchor="middle" font-size="8" fill="#15803d">REQ2</text>
  <rect x="340" y="101" width="50" height="14" rx="2" fill="#86efac"/>
  <text x="365" y="112" text-anchor="middle" font-size="8" fill="#15803d">RESP2</text>
  <text x="360" y="136" text-anchor="middle" font-size="8" fill="#c2410c">⚠ HOL: REQ3 waits</text>

  <!-- HTTP/2: 1 connection, parallel streams -->
  <rect x="510" y="45" width="180" height="100" rx="3" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="600" y="60" text-anchor="middle" font-size="9" fill="#7c3aed">1 TCP connection</text>
  <rect x="520" y="67" width="150" height="12" rx="2" fill="#c4b5fd"/>
  <text x="595" y="77" text-anchor="middle" font-size="8" fill="#4c1d95">Stream 1: REQ1 ──── RESP1</text>
  <rect x="520" y="83" width="150" height="12" rx="2" fill="#ddd6fe"/>
  <text x="595" y="93" text-anchor="middle" font-size="8" fill="#4c1d95">Stream 3: REQ2 ──── RESP2</text>
  <rect x="520" y="99" width="150" height="12" rx="2" fill="#ede9fe"/>
  <text x="595" y="109" text-anchor="middle" font-size="8" fill="#4c1d95">Stream 5: REQ3 ──── RESP3</text>
  <text x="600" y="133" text-anchor="middle" font-size="8" fill="#15803d">✓ Parallel, no HOL</text>
</svg>
</div>
```

- [ ] Verify: 3 column layout renders correctly in browser

---

## Task 4 — `http-deep-dive.html`: Sections s4–s7

- [ ] **s4 — HTTP Methods & Semantics**
  - Table: Method | Safe | Idempotent | Request Body | Description — for GET/HEAD/POST/PUT/DELETE/PATCH/OPTIONS/CONNECT/TRACE
  - Box blue "Safe vs Idempotent": Safe = no side effects (GET, HEAD, OPTIONS). Idempotent = same result if repeated (GET, PUT, DELETE)
  - Sub-section "PATCH vs PUT": PUT thay toàn bộ resource, PATCH thay một phần
  - Code block: ví dụ PATCH request với Content-Type: application/json-patch+json

- [ ] **s5 — HTTP Headers & Message Structure** + SVG 3
  - **SVG 3 — Request/Response Anatomy** (viewBox="0 0 700 220"):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 220" xmlns="http://www.w3.org/2000/svg">
  <!-- Request side -->
  <text x="175" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="#0369a1">HTTP Request</text>
  <rect x="20" y="28" width="310" height="30" rx="4" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="175" y="49" text-anchor="middle" font-size="10" font-weight="700" fill="#0369a1">Request Line: GET /path?q=1 HTTP/1.1</text>

  <rect x="20" y="62" width="310" height="18" rx="2" fill="#f0f9ff" stroke="#bae6fd"/>
  <text x="175" y="75" text-anchor="middle" font-size="9" fill="#0369a1">Host: example.com</text>
  <rect x="20" y="82" width="310" height="18" rx="2" fill="#f0f9ff" stroke="#bae6fd"/>
  <text x="175" y="95" text-anchor="middle" font-size="9" fill="#0369a1">Accept: text/html, application/json</text>
  <rect x="20" y="102" width="310" height="18" rx="2" fill="#f0f9ff" stroke="#bae6fd"/>
  <text x="175" y="115" text-anchor="middle" font-size="9" fill="#0369a1">Authorization: Bearer &lt;token&gt;</text>
  <rect x="20" y="122" width="310" height="18" rx="2" fill="#f0f9ff" stroke="#bae6fd"/>
  <text x="175" y="135" text-anchor="middle" font-size="9" fill="#0369a1">User-Agent: Mozilla/5.0 …</text>
  <text x="30" y="152" font-size="9" fill="#94a3b8" font-style="italic">(blank line — CRLF)</text>
  <rect x="20" y="160" width="310" height="40" rx="2" fill="#f8fafc" stroke="#e2e8f0" stroke-dasharray="4 2"/>
  <text x="175" y="185" text-anchor="middle" font-size="9" fill="#94a3b8">Request Body (POST/PUT/PATCH)</text>

  <text x="30" y="42" font-size="8" fill="#7c3aed">start-line</text>
  <text x="30" y="72" font-size="8" fill="#0369a1">headers</text>

  <!-- Response side -->
  <text x="525" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">HTTP Response</text>
  <rect x="370" y="28" width="310" height="30" rx="4" fill="#dcfce7" stroke="#86efac"/>
  <text x="525" y="49" text-anchor="middle" font-size="10" font-weight="700" fill="#15803d">Status Line: HTTP/1.1 200 OK</text>

  <rect x="370" y="62" width="310" height="18" rx="2" fill="#f0fdf4" stroke="#bbf7d0"/>
  <text x="525" y="75" text-anchor="middle" font-size="9" fill="#15803d">Content-Type: application/json</text>
  <rect x="370" y="82" width="310" height="18" rx="2" fill="#f0fdf4" stroke="#bbf7d0"/>
  <text x="525" y="95" text-anchor="middle" font-size="9" fill="#15803d">Content-Length: 342</text>
  <rect x="370" y="102" width="310" height="18" rx="2" fill="#f0fdf4" stroke="#bbf7d0"/>
  <text x="525" y="115" text-anchor="middle" font-size="9" fill="#15803d">Cache-Control: max-age=3600</text>
  <rect x="370" y="122" width="310" height="18" rx="2" fill="#f0fdf4" stroke="#bbf7d0"/>
  <text x="525" y="135" text-anchor="middle" font-size="9" fill="#15803d">Set-Cookie: session=abc; HttpOnly</text>
  <text x="380" y="152" font-size="9" fill="#94a3b8" font-style="italic">(blank line — CRLF)</text>
  <rect x="370" y="160" width="310" height="40" rx="2" fill="#f0fdf4" stroke="#86efac"/>
  <text x="525" y="185" text-anchor="middle" font-size="9" fill="#15803d">{"id":1,"name":"Alice"} (body)</text>
</svg>
</div>
```

  - Sub-section "Request Headers phổ biến": table Host/Accept/Accept-Encoding/Authorization/Cookie/Content-Type/Content-Length/If-None-Match/Cache-Control/Origin
  - Sub-section "Response Headers phổ biến": table Content-Type/Content-Length/Content-Encoding/Set-Cookie/Location/ETag/Last-Modified/Cache-Control/Access-Control-Allow-Origin/Strict-Transport-Security

- [ ] **s6 — Status Codes**
  - 5 sub-sections (1xx/2xx/3xx/4xx/5xx), each with a table of common codes
  - Key codes to explain: 200/201/204, 301/302/304/307/308, 400/401/403/404/405/409/422/429, 500/502/503/504
  - Box yellow "301 vs 302 vs 307 vs 308": permanent vs temporary, method preservation
  - Box orange "401 vs 403": unauthenticated vs unauthorized

- [ ] **s7 — URL & Content Negotiation**
  - Diagram (pure CSS/HTML, no SVG needed): URL anatomy — `https://user:pass@example.com:8080/path?k=v#frag` with labeled parts
  - Sub-section "Percent Encoding": `%20` = space, `%2F` = /, RFC 3986 unreserved chars
  - Sub-section "Content Negotiation": Accept/Content-Type, Accept-Language, Accept-Encoding (gzip/br/zstd), q-values (`Accept: text/html;q=0.9, application/json;q=0.8`)
  - Sub-section "Vary header": tells caches which request headers vary the response

---

## Task 5 — `http-deep-dive.html`: Sections s8–s11

- [ ] **s8 — Caching**
  - Sub-section "Cache-Control directives" — table: max-age, s-maxage, no-cache, no-store, must-revalidate, stale-while-revalidate, stale-if-error, private, public, immutable
  - Sub-section "Validation": ETag + If-None-Match (strong/weak ETags), Last-Modified + If-Modified-Since → 304 Not Modified
  - Sub-section "Cache layers": Browser cache → Service Worker → CDN edge → Origin
  - Code block: `Cache-Control: max-age=86400, stale-while-revalidate=3600`
  - Box blue "no-cache vs no-store": no-cache = must revalidate before using; no-store = never store

- [ ] **s9 — Cookies & Sessions**
  - Code block: `Set-Cookie: session=abc123; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=3600`
  - Table: Cookie attribute | Effect | Security implication — HttpOnly/Secure/SameSite(None/Lax/Strict)/Domain/Path/Max-Age/Expires
  - Sub-section "SameSite": Lax (default) = gửi trong same-site navigation; Strict = same-site only; None = cross-site (requires Secure)
  - Sub-section "Session patterns": server-side sessions vs JWT (stateless), session fixation attack

- [ ] **s10 — CORS**
  - Sub-section "Same-Origin Policy": scheme + host + port phải giống nhau
  - Sub-section "Simple Requests": GET/HEAD/POST với safe content-types → no preflight
  - Sub-section "Preflighted Requests": OPTIONS preflight → Access-Control-Allow-* → actual request
  - Code block: preflight request + response headers example
  - Table: CORS header | Meaning — Access-Control-Allow-Origin/Methods/Headers/Credentials/Expose-Headers/Max-Age
  - Box red "Pitfall": `Access-Control-Allow-Origin: *` không hoạt động với `credentials: 'include'`

- [ ] **s11 — HTTPS & PKI**
  - Sub-section "Why HTTPS": snooping/tampering on HTTP, 3 guarantees: confidentiality/integrity/authenticity
  - Sub-section "Certificates (X.509)": Subject/Issuer/Serial/ValidFrom-To/Public Key/Signature, CN vs SAN
  - Sub-section "CA Chain": Root CA → Intermediate CA → Leaf cert, trust store, certificate pinning (HPKP deprecated → Expect-CT)
  - Sub-section "Certificate Transparency": CT logs (RFC 6962), SCTs, Chrome requirement
  - Sub-section "SNI": Server Name Indication — TLS extension, giải quyết vấn đề 1 IP nhiều certs

---

## Task 6 — `http-deep-dive.html`: Section s12 + SVG 4 (TLS 1.3 Handshake)

- [ ] **s12 — TLS Handshake** + SVG 4

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr-tls" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#64748b"/>
    </marker>
    <marker id="arr-tls-green" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#22c55e"/>
    </marker>
  </defs>
  <!-- Actors -->
  <rect x="30" y="10" width="100" height="30" rx="5" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="80" y="30" text-anchor="middle" font-size="11" font-weight="700" fill="#0369a1">Client</text>
  <rect x="570" y="10" width="100" height="30" rx="5" fill="#dcfce7" stroke="#86efac"/>
  <text x="620" y="30" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">Server</text>

  <!-- Lifelines -->
  <line x1="80" y1="40" x2="80" y2="390" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="4 2"/>
  <line x1="620" y1="40" x2="620" y2="390" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="4 2"/>

  <!-- Divider bands -->
  <rect x="0" y="55" width="700" height="18" fill="#f1f5f9"/>
  <text x="350" y="67" text-anchor="middle" font-size="9" font-weight="700" fill="#64748b">TLS 1.3 — 1-RTT Handshake</text>

  <!-- Step 1: ClientHello -->
  <line x1="80" y1="90" x2="612" y2="90" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-tls)"/>
  <text x="350" y="85" text-anchor="middle" font-size="9" font-weight="700" fill="#0369a1">ClientHello</text>
  <text x="350" y="100" text-anchor="middle" font-size="8" fill="#64748b">supported cipher suites, key_share (ECDHE public key), SNI</text>

  <!-- Step 2: ServerHello + cert + Finished -->
  <line x1="620" y1="130" x2="88" y2="130" stroke="#15803d" stroke-width="1.5" marker-end="url(#arr-tls)"/>
  <text x="350" y="125" text-anchor="middle" font-size="9" font-weight="700" fill="#15803d">ServerHello</text>
  <text x="350" y="140" text-anchor="middle" font-size="8" fill="#64748b">selected cipher, key_share (server ECDHE public key)</text>

  <line x1="620" y1="165" x2="88" y2="165" stroke="#15803d" stroke-width="1.5" marker-end="url(#arr-tls)"/>
  <text x="350" y="160" text-anchor="middle" font-size="9" fill="#15803d">{EncryptedExtensions + Certificate + CertificateVerify}</text>
  <text x="350" y="175" text-anchor="middle" font-size="8" fill="#64748b">server cert + signature over handshake (private key proof)</text>

  <line x1="620" y1="200" x2="88" y2="200" stroke="#22c55e" stroke-width="2" marker-end="url(#arr-tls-green)"/>
  <text x="350" y="195" text-anchor="middle" font-size="9" font-weight="700" fill="#15803d">{Finished}</text>
  <text x="350" y="210" text-anchor="middle" font-size="8" fill="#64748b">HMAC over handshake transcript — keys established ✓</text>

  <!-- Step 3: Client Finished -->
  <line x1="80" y1="240" x2="612" y2="240" stroke="#22c55e" stroke-width="2" marker-end="url(#arr-tls)"/>
  <text x="350" y="235" text-anchor="middle" font-size="9" font-weight="700" fill="#0369a1">{Finished}</text>
  <text x="350" y="250" text-anchor="middle" font-size="8" fill="#64748b">client Finished — both sides verified</text>

  <!-- Application Data -->
  <rect x="0" y="260" width="700" height="18" fill="#f0fdf4"/>
  <text x="350" y="272" text-anchor="middle" font-size="9" font-weight="700" fill="#15803d">Application Data (encrypted AEAD)</text>

  <line x1="80" y1="290" x2="612" y2="290" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr-tls)"/>
  <text x="350" y="285" text-anchor="middle" font-size="9" fill="#64748b">GET / HTTP/2 (encrypted)</text>
  <line x1="620" y1="315" x2="88" y2="315" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr-tls)"/>
  <text x="350" y="310" text-anchor="middle" font-size="9" fill="#64748b">200 OK + body (encrypted)</text>

  <!-- 0-RTT note -->
  <rect x="0" y="340" width="700" height="18" fill="#fffbeb"/>
  <text x="350" y="352" text-anchor="middle" font-size="9" font-weight="700" fill="#92400e">0-RTT (Session Resumption) — Early Data</text>
  <line x1="80" y1="370" x2="612" y2="370" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#arr-tls)"/>
  <text x="350" y="365" text-anchor="middle" font-size="9" fill="#92400e">ClientHello + early_data (PSK) — replay risk ⚠</text>

  <!-- RTT bracket -->
  <line x1="665" y1="90" x2="665" y2="240" stroke="#94a3b8" stroke-width="1"/>
  <line x1="660" y1="90" x2="670" y2="90" stroke="#94a3b8" stroke-width="1"/>
  <line x1="660" y1="240" x2="670" y2="240" stroke="#94a3b8" stroke-width="1"/>
  <text x="680" y="170" font-size="9" fill="#94a3b8" transform="rotate(90,680,170)">1 RTT</text>
</svg>
</div>
```

  - After SVG: explain TLS 1.2 vs 1.3 differences (2-RTT vs 1-RTT, removed cipher suites, ECDHE mandatory)
  - Code block: cipher suite format — `TLS_AES_128_GCM_SHA256` (TLS 1.3) vs `TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256` (TLS 1.2)
  - Box blue "Key Exchange": ECDHE — Diffie-Hellman trên đường cong elliptic, perfect forward secrecy
  - Box orange "0-RTT Risks": replay attacks — `early_data` không nên dùng cho non-idempotent operations

---

## Task 7 — `http-deep-dive.html`: Sections s13–s14 + SVG 5 (Frame Layout)

- [ ] **s13 — HTTP/2 Binary Framing** + SVG 5

  SVG 5 — Frame layout (viewBox="0 0 700 200", bit-level rows):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Row labels -->
  <text x="55" y="58" text-anchor="end" font-size="9" fill="#64748b">Byte 0–2</text>
  <text x="55" y="98" text-anchor="end" font-size="9" fill="#64748b">Byte 3</text>
  <text x="55" y="138" text-anchor="end" font-size="9" fill="#64748b">Byte 4</text>
  <text x="55" y="178" text-anchor="end" font-size="9" fill="#64748b">Byte 5–8</text>

  <!-- Row 1: Length (24 bits) -->
  <rect x="60" y="42" width="620" height="26" rx="3" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="370" y="60" text-anchor="middle" font-size="10" font-weight="700" fill="#0369a1">Length (24 bits) — payload size in bytes, max 16,777,215</text>

  <!-- Row 2: Type (8 bits) -->
  <rect x="60" y="82" width="210" height="26" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="165" y="100" text-anchor="middle" font-size="10" font-weight="700" fill="#15803d">Type (8 bits)</text>
  <!-- Flags (8 bits) -->
  <rect x="278" y="82" width="210" height="26" rx="3" fill="#fef3c7" stroke="#fde68a"/>
  <text x="383" y="100" text-anchor="middle" font-size="10" font-weight="700" fill="#92400e">Flags (8 bits)</text>

  <!-- Row 3: Reserved (1) + Stream ID (31) -->
  <rect x="60" y="122" width="25" height="26" rx="2" fill="#f1f5f9" stroke="#e2e8f0"/>
  <text x="72" y="140" text-anchor="middle" font-size="8" fill="#94a3b8">R</text>
  <rect x="89" y="122" width="591" height="26" rx="3" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="384" y="140" text-anchor="middle" font-size="10" font-weight="700" fill="#7c3aed">Stream Identifier (31 bits) — 0 = connection-level</text>

  <!-- Row 4: Payload -->
  <rect x="60" y="162" width="620" height="26" rx="3" fill="#f8fafc" stroke="#e2e8f0" stroke-dasharray="5 3"/>
  <text x="370" y="180" text-anchor="middle" font-size="10" fill="#64748b">Payload (variable — frame type specific)</text>

  <!-- Bit ruler -->
  <text x="60" y="32" font-size="8" fill="#94a3b8">0</text>
  <text x="337" y="32" font-size="8" fill="#94a3b8">bit 15</text>
  <text x="670" y="32" font-size="8" fill="#94a3b8">31</text>
</svg>
</div>
```

  - Table: Frame Type | Hex | Mô tả — DATA(0x0)/HEADERS(0x1)/PRIORITY(0x2)/RST_STREAM(0x3)/SETTINGS(0x4)/PUSH_PROMISE(0x5)/PING(0x6)/GOAWAY(0x7)/WINDOW_UPDATE(0x8)/CONTINUATION(0x9)
  - Code block: SETTINGS frame example (HEADER_TABLE_SIZE, ENABLE_PUSH, MAX_CONCURRENT_STREAMS, INITIAL_WINDOW_SIZE)

- [ ] **s14 — Streams & Multiplexing**
  - Sub-section "Stream States": idle → open → half-closed(local) → half-closed(remote) → closed. Also: reserved(local/remote)
  - Sub-section "Stream IDs": client = odd (1,3,5…), server = even (2,4,6…), 0 = connection-level frames
  - Sub-section "Concurrent Streams": `SETTINGS_MAX_CONCURRENT_STREAMS` (default unlimited, typically 100)
  - Sub-section "HOL Blocking Fix": HTTP/2 fixes HTTP-layer HOL but TCP-layer HOL remains (solved by HTTP/3 QUIC)
  - Sub-section "RST_STREAM vs GOAWAY": RST_STREAM = cancel single stream; GOAWAY = close connection gracefully with last processed stream ID

---

## Task 8 — `http-deep-dive.html`: Sections s15–s16 + SVG 6 (HPACK)

- [ ] **s15 — HPACK Header Compression** + SVG 6
  - Sub-section "Static Table": 61 pre-defined entries (`:authority`, `:method GET`, `:method POST`, `:path /`, `content-type`, etc.), indexed 1–61
  - Sub-section "Dynamic Table": FIFO eviction, max size negotiated via `SETTINGS_HEADER_TABLE_SIZE`
  - Sub-section "Encoding types": Indexed (1 byte), Literal with Incremental Indexing, Literal without Indexing, Literal Never Indexed (sensitive headers — auth tokens)
  - Sub-section "Huffman Coding": entropy coding từ HTTP/1.1 traffic statistics, ~30% reduction

  SVG 6 — HPACK lookup (viewBox="0 0 700 180"):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 180" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr-hp" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#64748b"/>
    </marker>
  </defs>
  <!-- Header field input -->
  <rect x="10" y="70" width="130" height="40" rx="5" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="75" y="86" text-anchor="middle" font-size="10" font-weight="700" fill="#0369a1">Header Field</text>
  <text x="75" y="101" text-anchor="middle" font-size="9" fill="#0369a1">e.g. content-type: json</text>
  <line x1="140" y1="90" x2="168" y2="90" stroke="#94a3b8" marker-end="url(#arr-hp)"/>

  <!-- Static table check -->
  <rect x="170" y="55" width="130" height="70" rx="5" fill="#dcfce7" stroke="#86efac"/>
  <text x="235" y="75" text-anchor="middle" font-size="10" font-weight="700" fill="#15803d">Static Table</text>
  <text x="235" y="90" text-anchor="middle" font-size="8" fill="#15803d">61 entries</text>
  <text x="235" y="105" text-anchor="middle" font-size="8" fill="#15803d">RFC-defined</text>
  <text x="235" y="118" text-anchor="middle" font-size="8" fill="#15803d">e.g. idx 31 = content-type</text>

  <!-- Dynamic table check -->
  <rect x="340" y="55" width="130" height="70" rx="5" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="405" y="75" text-anchor="middle" font-size="10" font-weight="700" fill="#7c3aed">Dynamic Table</text>
  <text x="405" y="90" text-anchor="middle" font-size="8" fill="#7c3aed">FIFO, per-connection</text>
  <text x="405" y="105" text-anchor="middle" font-size="8" fill="#7c3aed">grows with new headers</text>
  <text x="405" y="118" text-anchor="middle" font-size="8" fill="#7c3aed">idx 62, 63, 64…</text>

  <line x1="300" y1="90" x2="338" y2="90" stroke="#94a3b8" marker-end="url(#arr-hp)"/>
  <text x="319" y="85" text-anchor="middle" font-size="8" fill="#94a3b8">miss</text>

  <!-- Output: indexed or literal -->
  <rect x="510" y="40" width="110" height="30" rx="4" fill="#dcfce7" stroke="#86efac"/>
  <text x="565" y="60" text-anchor="middle" font-size="9" font-weight="700" fill="#15803d">Indexed (1 byte)</text>

  <rect x="510" y="80" width="110" height="30" rx="4" fill="#fef3c7" stroke="#fde68a"/>
  <text x="565" y="100" text-anchor="middle" font-size="9" font-weight="700" fill="#92400e">Literal + index</text>

  <rect x="510" y="120" width="110" height="30" rx="4" fill="#fef2f2" stroke="#fca5a5"/>
  <text x="565" y="140" text-anchor="middle" font-size="9" font-weight="700" fill="#dc2626">Never Index</text>
  <text x="565" y="152" text-anchor="middle" font-size="8" fill="#dc2626">(Authorization)</text>

  <line x1="300" y1="90" x2="508" y2="55" stroke="#15803d" stroke-dasharray="3 2" marker-end="url(#arr-hp)"/>
  <text x="400" y="68" text-anchor="middle" font-size="8" fill="#15803d">hit</text>
  <line x1="470" y1="90" x2="508" y2="95" stroke="#92400e" marker-end="url(#arr-hp)"/>
  <line x1="405" y1="125" x2="508" y2="135" stroke="#dc2626" stroke-dasharray="3 2" marker-end="url(#arr-hp)"/>
</svg>
</div>
```

- [ ] **s16 — Flow Control & Priorities**
  - Sub-section "Flow Control": per-stream window + per-connection window, default 65,535 bytes, WINDOW_UPDATE frame tăng window
  - Sub-section "Priority (deprecated in HTTP/3)": PRIORITY frame, dependency tree, weight (1–256), priority inversion
  - Sub-section "SETTINGS_INITIAL_WINDOW_SIZE": ảnh hưởng tất cả streams mới
  - Box orange "Deprecation": RFC 9113 loại bỏ priority tree, thay bằng Extensible Priorities (RFC 9218)
  - Code block: WINDOW_UPDATE example

---

## Task 9 — `http-deep-dive.html`: Sections s17–s18 + SVG 7 (QUIC Stack)

- [ ] **s17 — Security Headers**
  - Table: Header | Example | Tác dụng — HSTS/CSP/X-Frame-Options/Referrer-Policy/Permissions-Policy/CORP/COOP/COEP/X-Content-Type-Options/X-XSS-Protection(deprecated)
  - Code block: full security headers example for nginx config
  - Box blue "HSTS Preload": chromium preload list, includeSubDomains requirement, hstspreload.org
  - Box red "CSP gotchas": `unsafe-inline` phủ nhận CSP, nonce-based CSP, `report-uri` vs `report-to`

- [ ] **s18 — HTTP/3 & QUIC** + SVG 7

  SVG 7 — QUIC vs TCP Stack (viewBox="0 0 700 240"):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 240" xmlns="http://www.w3.org/2000/svg">
  <!-- Left: TCP+TLS+H2 -->
  <text x="175" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="#0369a1">HTTP/2 over TLS/TCP</text>
  <rect x="30" y="28" width="290" height="34" rx="4" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="175" y="50" text-anchor="middle" font-size="10" font-weight="700" fill="#0369a1">HTTP/2 Application</text>

  <rect x="30" y="66" width="290" height="34" rx="4" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="175" y="88" text-anchor="middle" font-size="10" font-weight="700" fill="#7c3aed">TLS 1.3</text>
  <text x="175" y="99" text-anchor="middle" font-size="8" fill="#7c3aed">separate handshake layer</text>

  <rect x="30" y="104" width="290" height="34" rx="4" fill="#dcfce7" stroke="#86efac"/>
  <text x="175" y="126" text-anchor="middle" font-size="10" font-weight="700" fill="#15803d">TCP</text>
  <text x="175" y="137" text-anchor="middle" font-size="8" fill="#15803d">stream-oriented, HOL blocking</text>

  <rect x="30" y="142" width="290" height="34" rx="4" fill="#fff7ed" stroke="#fdba74"/>
  <text x="175" y="164" text-anchor="middle" font-size="10" font-weight="700" fill="#c2410c">IP</text>

  <!-- Latency label -->
  <rect x="30" y="184" width="290" height="24" rx="4" fill="#fef2f2" stroke="#fca5a5"/>
  <text x="175" y="200" text-anchor="middle" font-size="9" fill="#dc2626">TCP 3-way + TLS 1-RTT = 2 RTT to first byte</text>

  <!-- Right: QUIC+H3 -->
  <text x="525" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="#7c3aed">HTTP/3 over QUIC</text>
  <rect x="380" y="28" width="290" height="34" rx="4" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="525" y="50" text-anchor="middle" font-size="10" font-weight="700" fill="#0369a1">HTTP/3 Application</text>

  <rect x="380" y="66" width="290" height="72" rx="4" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="525" y="88" text-anchor="middle" font-size="10" font-weight="700" fill="#7c3aed">QUIC (UDP)</text>
  <text x="525" y="103" text-anchor="middle" font-size="8" fill="#7c3aed">TLS 1.3 integrated</text>
  <text x="525" y="115" text-anchor="middle" font-size="8" fill="#7c3aed">independent streams (no HOL)</text>
  <text x="525" y="127" text-anchor="middle" font-size="8" fill="#7c3aed">connection migration (CID)</text>

  <rect x="380" y="142" width="290" height="34" rx="4" fill="#fff7ed" stroke="#fdba74"/>
  <text x="525" y="164" text-anchor="middle" font-size="10" font-weight="700" fill="#c2410c">UDP + IP</text>

  <!-- Latency label -->
  <rect x="380" y="184" width="290" height="24" rx="4" fill="#dcfce7" stroke="#86efac"/>
  <text x="525" y="200" text-anchor="middle" font-size="9" fill="#15803d">0-RTT resumption / 1-RTT new connection</text>

  <!-- Comparison arrow -->
  <text x="350" y="110" text-anchor="middle" font-size="18" fill="#94a3b8">→</text>
  <text x="350" y="125" text-anchor="middle" font-size="8" fill="#94a3b8">evolution</text>
</svg>
</div>
```

  - Sub-section "QUIC features": multiplexed streams over UDP, 0-RTT, connection migration via Connection ID (phone → WiFi không reset connection), built-in congestion control
  - Sub-section "HTTP/3 differences": no HPACK → QPACK (QUIC-aware), no server push (deprecated), no priority tree → Extensible Priorities
  - Table: HTTP/1.1 vs HTTP/2 vs HTTP/3 comparison — transport/latency/compression/multiplexing/HOL/server push

- [ ] Verify: open `http-deep-dive.html` in browser, scroll all 18 sections, check sidebar active state, all 7 SVGs render without overflow

---

## Task 10 — `grpc-deep-dive.html`: Shell + CSS + Sidebar

**Files:** Create `docs/grpc-deep-dive.html`

- [ ] Create file with same CSS template, title "gRPC Deep Dive"
- [ ] Sidebar with 4 nav groups:

```html
<nav class="sidebar">
  <div class="sidebar-title">gRPC Deep Dive</div>
  <div class="nav-group-label">Nền tảng</div>
  <a class="nav-item" href="#s1"><span class="nav-num">01</span>gRPC là gì?</a>
  <a class="nav-item" href="#s2"><span class="nav-num">02</span>Proto3 Syntax</a>
  <a class="nav-item" href="#s3"><span class="nav-num">03</span>Wire Format</a>
  <a class="nav-item" href="#s4"><span class="nav-num">04</span>Service Definition</a>
  <a class="nav-item" href="#s5"><span class="nav-num">05</span>Code Generation</a>
  <div class="nav-divider"></div>
  <div class="nav-group-label">Communication Patterns</div>
  <a class="nav-item" href="#s6"><span class="nav-num">06</span>Unary RPC</a>
  <a class="nav-item" href="#s7"><span class="nav-num">07</span>Server Streaming</a>
  <a class="nav-item" href="#s8"><span class="nav-num">08</span>Client Streaming</a>
  <a class="nav-item" href="#s9"><span class="nav-num">09</span>Bidirectional Streaming</a>
  <div class="nav-divider"></div>
  <div class="nav-group-label">Runtime</div>
  <a class="nav-item" href="#s10"><span class="nav-num">10</span>Channels &amp; Stubs</a>
  <a class="nav-item" href="#s11"><span class="nav-num">11</span>Metadata</a>
  <a class="nav-item" href="#s12"><span class="nav-num">12</span>Error Handling</a>
  <a class="nav-item" href="#s13"><span class="nav-num">13</span>Deadlines &amp; Cancellation</a>
  <div class="nav-divider"></div>
  <div class="nav-group-label">Production</div>
  <a class="nav-item" href="#s14"><span class="nav-num">14</span>Interceptors</a>
  <a class="nav-item" href="#s15"><span class="nav-num">15</span>Load Balancing</a>
  <a class="nav-item" href="#s16"><span class="nav-num">16</span>gRPC-Web &amp; Production</a>
</nav>
```

- [ ] Add `page-title` "gRPC Deep Dive" and `page-sub` subtitle
- [ ] Add IntersectionObserver script
- [ ] Open in browser — verify layout

---

## Task 11 — `grpc-deep-dive.html`: SVG 1 (Stack) + Sections s1–s2

- [ ] **SVG 1 — gRPC Stack** (place after page-sub, before s1):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 120" xmlns="http://www.w3.org/2000/svg">
  <!-- Vertical stack left side -->
  <rect x="30" y="10" width="280" height="22" rx="3" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="170" y="25" text-anchor="middle" font-size="10" font-weight="700" fill="#0369a1">Application Code (any language)</text>
  <rect x="30" y="36" width="280" height="22" rx="3" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="170" y="51" text-anchor="middle" font-size="10" font-weight="700" fill="#7c3aed">gRPC Framework (stubs + runtime)</text>
  <rect x="30" y="62" width="280" height="22" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="170" y="77" text-anchor="middle" font-size="10" font-weight="700" fill="#15803d">HTTP/2 Transport</text>
  <rect x="30" y="88" width="280" height="22" rx="3" fill="#fff7ed" stroke="#fdba74"/>
  <text x="170" y="103" text-anchor="middle" font-size="10" font-weight="700" fill="#c2410c">TLS / TCP / IP</text>

  <!-- Right side: Protobuf -->
  <rect x="390" y="36" width="280" height="22" rx="3" fill="#fef3c7" stroke="#fde68a"/>
  <text x="530" y="51" text-anchor="middle" font-size="10" font-weight="700" fill="#92400e">Protocol Buffers (serialization)</text>
  <rect x="390" y="62" width="280" height="22" rx="3" fill="#fef3c7" stroke="#fde68a"/>
  <text x="530" y="77" text-anchor="middle" font-size="10" fill="#92400e">.proto → generated code</text>

  <!-- Arrows -->
  <line x1="310" y1="47" x2="388" y2="47" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3 2"/>
  <text x="349" y="43" text-anchor="middle" font-size="8" fill="#94a3b8">encode/decode</text>

  <!-- Bracket labels -->
  <text x="350" y="15" text-anchor="middle" font-size="9" fill="#64748b">Server &amp; Client both</text>
</svg>
</div>
```

- [ ] **s1 — gRPC là gì?**
  - Paragraph: Google RPC framework (2015 internal "Stubby" → 2016 open source as gRPC), HTTP/2 transport, Protobuf serialization, polyglot (10+ languages)
  - Box blue "So với REST": gRPC = strongly-typed contracts, binary wire format, streaming built-in, code generation; REST = human-readable, flexible, browser-native
  - Table: Feature | gRPC | REST — Schema/Transport/Payload/Streaming/Browser support/Code gen/Performance
  - Sub-section "Use cases": microservices inter-service communication, mobile clients (bandwidth), real-time streaming (telemetry, chat), internal APIs

- [ ] **s2 — Protocol Buffers — Proto3 Syntax**
  - Code block: full .proto example with message, enum, map, oneof, repeated, import, package, option go_package
  - Table: Scalar types — double/float/int32/int64/uint32/uint64/sint32/sint64/fixed32/fixed64/sfixed32/sfixed64/bool/string/bytes
  - Sub-section "Field Numbers": 1–15 = 1 byte tag, 16–2047 = 2 bytes; never reuse a number; reserved statement
  - Sub-section "Default Values": 0 for numerics, "" for string, false for bool, empty for repeated — no null in proto3
  - Box orange "Proto3 vs Proto2": no required fields, no default values in syntax, optional keyword re-added in proto3 optional

---

## Task 12 — `grpc-deep-dive.html`: Section s3 + SVG 2 (Wire Format)

- [ ] **s3 — Protobuf Wire Format** + SVG 2
  - Sub-section "Wire Types": table wire_type | Meaning | Used for — 0 Varint/1 64-bit/2 LEN/5 32-bit
  - Sub-section "Field Tag": `(field_number << 3) | wire_type` — example: field 1, wire type 2 → tag byte = 0x0A
  - Sub-section "Varint Encoding": MSB = continuation bit, 7 bits of data per byte, little-endian group ordering. Example: 300 = 0xAC 0x02

  SVG 2 — Varint + field tag (viewBox="0 0 700 200"):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Varint section -->
  <text x="10" y="18" font-size="11" font-weight="700" fill="#0369a1">Varint Encoding — number 300 (0x12C)</text>

  <!-- Byte 1: 1010 1100 = 0xAC -->
  <rect x="10" y="28" width="20" height="28" rx="2" fill="#fef3c7" stroke="#fde68a"/>
  <text x="20" y="46" text-anchor="middle" font-size="9" font-weight="700" fill="#c2410c">1</text>
  <rect x="32" y="28" width="140" height="28" rx="2" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="102" y="46" text-anchor="middle" font-size="9" fill="#0369a1">010 1100  (low 7 bits of 300)</text>
  <text x="20" y="68" text-anchor="middle" font-size="8" fill="#c2410c">more</text>

  <!-- Byte 2: 0000 0010 = 0x02 -->
  <rect x="210" y="28" width="20" height="28" rx="2" fill="#dcfce7" stroke="#86efac"/>
  <text x="220" y="46" text-anchor="middle" font-size="9" font-weight="700" fill="#15803d">0</text>
  <rect x="232" y="28" width="140" height="28" rx="2" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="302" y="46" text-anchor="middle" font-size="9" fill="#0369a1">000 0010  (high bits of 300)</text>
  <text x="220" y="68" text-anchor="middle" font-size="8" fill="#15803d">last</text>

  <text x="400" y="42" font-size="9" fill="#64748b">= 0xAC 0x02 (2 bytes for 300)</text>
  <text x="400" y="55" font-size="9" fill="#64748b">vs 4 bytes for int32 in fixed encoding</text>

  <!-- Field tag section -->
  <text x="10" y="100" font-size="11" font-weight="700" fill="#7c3aed">Field Tag Encoding</text>
  <text x="10" y="115" font-size="9" fill="#64748b">field_number=2, wire_type=2 (LEN) → tag = (2 &lt;&lt; 3) | 2 = 0x12</text>

  <!-- Bit layout for tag -->
  <rect x="10" y="125" width="400" height="28" rx="3" fill="#faf5ff" stroke="#c4b5fd"/>
  <!-- bit groups -->
  <rect x="12" y="127" width="280" height="24" rx="2" fill="#ede9fe"/>
  <text x="152" y="143" text-anchor="middle" font-size="9" fill="#7c3aed">field_number (bits 3–31)</text>
  <rect x="296" y="127" width="112" height="24" rx="2" fill="#fef3c7"/>
  <text x="352" y="143" text-anchor="middle" font-size="9" fill="#c2410c">wire_type (bits 0–2)</text>

  <!-- Wire type table -->
  <text x="10" y="175" font-size="11" font-weight="700" fill="#15803d">Wire Types</text>
  <rect x="10" y="183" width="55" height="15" rx="2" fill="#dcfce7"/>
  <text x="37" y="194" text-anchor="middle" font-size="8" fill="#15803d">0: Varint</text>
  <rect x="70" y="183" width="55" height="15" rx="2" fill="#e0f2fe"/>
  <text x="97" y="194" text-anchor="middle" font-size="8" fill="#0369a1">1: 64-bit</text>
  <rect x="130" y="183" width="55" height="15" rx="2" fill="#faf5ff"/>
  <text x="157" y="194" text-anchor="middle" font-size="8" fill="#7c3aed">2: LEN</text>
  <rect x="190" y="183" width="55" height="15" rx="2" fill="#fef3c7"/>
  <text x="217" y="194" text-anchor="middle" font-size="8" fill="#c2410c">5: 32-bit</text>
  <text x="260" y="194" font-size="8" fill="#94a3b8">(types 3 &amp; 4 deprecated)</text>
</svg>
</div>
```

  - Code block: hexdump của một message proto3 với explanation — e.g. `{name: "Alice", id: 1}` → `0a 05 41 6c 69 63 65 10 01`

---

## Task 13 — `grpc-deep-dive.html`: Sections s4–s5

- [ ] **s4 — Service Definition**
  - Code block: complete .proto service definition với tất cả 4 RPC patterns:
    ```protobuf
    service ChatService {
      rpc SendMessage (MessageRequest) returns (MessageResponse);            // unary
      rpc SubscribeMessages (SubscribeRequest) returns (stream Message);     // server streaming
      rpc UploadMessages (stream MessageRequest) returns (UploadSummary);   // client streaming
      rpc Chat (stream Message) returns (stream Message);                   // bidirectional
    }
    ```
  - Sub-section "Well-Known Types": google.protobuf.Timestamp, Duration, Any, Empty, Struct, Value, FieldMask
  - Sub-section "Best Practices": field number 1–15 cho hot fields, không bao giờ reuse numbers, use FieldMask cho partial updates, version in package name (v1alpha1/v1)
  - Box yellow "Backward Compatibility": thêm field mới = OK (default values); rename = breaking (field number unchanged); change type = breaking

- [ ] **s5 — Code Generation**
  - Code block: `protoc` command với --go_out, --go-grpc_out flags
  - Code block: buf.gen.yaml example
  - Sub-section "Generated code structure": message structs/classes, Marshal/Unmarshal, XxxServiceClient interface, XxxServiceServer interface, RegisterXxxServiceServer()
  - Sub-section "buf vs protoc": buf có breaking change detection, lint, remote BSR (Buf Schema Registry)
  - Code block: `buf breaking --against '.git#branch=main'`
  - Box blue "Language support": Go, Java, Python, C++, C#, Ruby, PHP, Dart, Kotlin, Swift, Node.js

---

## Task 14 — `grpc-deep-dive.html`: Sections s6–s9 + SVG 3 (4 Patterns)

- [ ] **SVG 3 — 4 Streaming Patterns** (viewBox="0 0 700 260", 4 panels):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 260" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr-g" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#64748b"/>
    </marker>
    <marker id="arr-blue" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#0ea5e9"/>
    </marker>
    <marker id="arr-green" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#22c55e"/>
    </marker>
  </defs>

  <!-- Panel headers (4 columns) -->
  <text x="88" y="14" text-anchor="middle" font-size="10" font-weight="700" fill="#0369a1">Unary</text>
  <text x="263" y="14" text-anchor="middle" font-size="10" font-weight="700" fill="#7c3aed">Server Streaming</text>
  <text x="438" y="14" text-anchor="middle" font-size="10" font-weight="700" fill="#15803d">Client Streaming</text>
  <text x="613" y="14" text-anchor="middle" font-size="10" font-weight="700" fill="#c2410c">Bidirectional</text>

  <!-- Column dividers -->
  <line x1="175" y1="18" x2="175" y2="260" stroke="#e2e8f0"/>
  <line x1="350" y1="18" x2="350" y2="260" stroke="#e2e8f0"/>
  <line x1="525" y1="18" x2="525" y2="260" stroke="#e2e8f0"/>

  <!-- Actor labels -->
  <text x="50" y="35" text-anchor="middle" font-size="9" fill="#0369a1">Client</text>
  <text x="125" y="35" text-anchor="middle" font-size="9" fill="#15803d">Server</text>
  <text x="225" y="35" text-anchor="middle" font-size="9" fill="#0369a1">Client</text>
  <text x="300" y="35" text-anchor="middle" font-size="9" fill="#15803d">Server</text>
  <text x="400" y="35" text-anchor="middle" font-size="9" fill="#0369a1">Client</text>
  <text x="475" y="35" text-anchor="middle" font-size="9" fill="#15803d">Server</text>
  <text x="575" y="35" text-anchor="middle" font-size="9" fill="#0369a1">Client</text>
  <text x="650" y="35" text-anchor="middle" font-size="9" fill="#15803d">Server</text>

  <!-- Lifelines -->
  <line x1="50" y1="40" x2="50" y2="255" stroke="#bae6fd" stroke-dasharray="3 2"/>
  <line x1="125" y1="40" x2="125" y2="255" stroke="#bbf7d0" stroke-dasharray="3 2"/>
  <line x1="225" y1="40" x2="225" y2="255" stroke="#bae6fd" stroke-dasharray="3 2"/>
  <line x1="300" y1="40" x2="300" y2="255" stroke="#bbf7d0" stroke-dasharray="3 2"/>
  <line x1="400" y1="40" x2="400" y2="255" stroke="#bae6fd" stroke-dasharray="3 2"/>
  <line x1="475" y1="40" x2="475" y2="255" stroke="#bbf7d0" stroke-dasharray="3 2"/>
  <line x1="575" y1="40" x2="575" y2="255" stroke="#bae6fd" stroke-dasharray="3 2"/>
  <line x1="650" y1="40" x2="650" y2="255" stroke="#bbf7d0" stroke-dasharray="3 2"/>

  <!-- Unary: 1 req → 1 resp -->
  <line x1="50" y1="70" x2="117" y2="70" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-blue)"/>
  <text x="83" y="65" text-anchor="middle" font-size="8" fill="#0369a1">Request</text>
  <line x1="125" y1="110" x2="58" y2="110" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <text x="83" y="105" text-anchor="middle" font-size="8" fill="#15803d">Response</text>

  <!-- Server Streaming: 1 req → N resp -->
  <line x1="225" y1="65" x2="292" y2="65" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-blue)"/>
  <text x="258" y="60" text-anchor="middle" font-size="8" fill="#0369a1">Request</text>
  <line x1="300" y1="90" x2="233" y2="90" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <text x="258" y="85" text-anchor="middle" font-size="8" fill="#15803d">Response 1</text>
  <line x1="300" y1="115" x2="233" y2="115" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <text x="258" y="110" text-anchor="middle" font-size="8" fill="#15803d">Response 2</text>
  <line x1="300" y1="140" x2="233" y2="140" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <text x="258" y="135" text-anchor="middle" font-size="8" fill="#15803d">Response N</text>
  <text x="258" y="160" text-anchor="middle" font-size="8" fill="#94a3b8">EOF (end of stream)</text>

  <!-- Client Streaming: N req → 1 resp -->
  <line x1="400" y1="65" x2="467" y2="65" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-blue)"/>
  <text x="433" y="60" text-anchor="middle" font-size="8" fill="#0369a1">Request 1</text>
  <line x1="400" y1="88" x2="467" y2="88" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-blue)"/>
  <text x="433" y="83" text-anchor="middle" font-size="8" fill="#0369a1">Request 2</text>
  <line x1="400" y1="111" x2="467" y2="111" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-blue)"/>
  <text x="433" y="106" text-anchor="middle" font-size="8" fill="#0369a1">Request N</text>
  <text x="433" y="128" text-anchor="middle" font-size="8" fill="#94a3b8">half-close</text>
  <line x1="475" y1="145" x2="408" y2="145" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <text x="433" y="140" text-anchor="middle" font-size="8" fill="#15803d">Response</text>

  <!-- Bidirectional -->
  <line x1="575" y1="60" x2="642" y2="60" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-blue)"/>
  <text x="608" y="55" text-anchor="middle" font-size="8" fill="#0369a1">Msg A1</text>
  <line x1="650" y1="80" x2="583" y2="80" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <text x="608" y="75" text-anchor="middle" font-size="8" fill="#15803d">Msg B1</text>
  <line x1="575" y1="100" x2="642" y2="100" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-blue)"/>
  <text x="608" y="95" text-anchor="middle" font-size="8" fill="#0369a1">Msg A2</text>
  <line x1="650" y1="120" x2="583" y2="120" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <text x="608" y="115" text-anchor="middle" font-size="8" fill="#15803d">Msg B2</text>
  <line x1="575" y1="140" x2="642" y2="140" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-blue)"/>
  <line x1="650" y1="155" x2="583" y2="155" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <text x="608" y="172" text-anchor="middle" font-size="8" fill="#94a3b8">independent streams</text>
</svg>
</div>
```

- [ ] **s6 — Unary RPC**: blocking stub example (Go), deadline, Status propagation, when to use
- [ ] **s7 — Server Streaming**: Go example (range over stream), use cases (live feed, large paginated result), EOF detection
- [ ] **s8 — Client Streaming**: Go example (send loop + CloseAndRecv), use cases (file upload, batch insert), half-close semantics
- [ ] **s9 — Bidirectional Streaming**: Go example (goroutines for send + recv), use cases (chat, real-time telemetry), flow control interaction

Each section: code block in Go showing client + server side (concise), box with use cases, note on error handling.

---

## Task 15 — `grpc-deep-dive.html`: Sections s10–s11 + SVG 4 (Channel Architecture)

- [ ] **s10 — Channels & Stubs** + SVG 4

  SVG 4 — Channel architecture (viewBox="0 0 700 160"):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 160" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr-ch" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#64748b"/>
    </marker>
  </defs>
  <!-- Stub -->
  <rect x="10" y="65" width="100" height="30" rx="5" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="60" y="84" text-anchor="middle" font-size="10" font-weight="700" fill="#0369a1">Stub</text>
  <line x1="110" y1="80" x2="148" y2="80" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-ch)"/>

  <!-- Channel -->
  <rect x="150" y="55" width="120" height="50" rx="5" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="210" y="75" text-anchor="middle" font-size="10" font-weight="700" fill="#7c3aed">Channel</text>
  <text x="210" y="90" text-anchor="middle" font-size="8" fill="#7c3aed">name resolution</text>
  <text x="210" y="102" text-anchor="middle" font-size="8" fill="#7c3aed">load balancing policy</text>
  <line x1="270" y1="70" x2="308" y2="50" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-ch)"/>
  <line x1="270" y1="80" x2="308" y2="80" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-ch)"/>
  <line x1="270" y1="90" x2="308" y2="110" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-ch)"/>

  <!-- Subchannels -->
  <rect x="310" y="30" width="120" height="30" rx="4" fill="#dcfce7" stroke="#86efac"/>
  <text x="370" y="50" text-anchor="middle" font-size="9" fill="#15803d">Subchannel 1 (READY)</text>
  <rect x="310" y="65" width="120" height="30" rx="4" fill="#dcfce7" stroke="#86efac"/>
  <text x="370" y="85" text-anchor="middle" font-size="9" fill="#15803d">Subchannel 2 (READY)</text>
  <rect x="310" y="100" width="120" height="30" rx="4" fill="#fef3c7" stroke="#fde68a"/>
  <text x="370" y="120" text-anchor="middle" font-size="9" fill="#92400e">Subchannel 3 (IDLE)</text>

  <!-- Servers -->
  <line x1="430" y1="45" x2="468" y2="45" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-ch)"/>
  <line x1="430" y1="80" x2="468" y2="80" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-ch)"/>
  <rect x="470" y="30" width="90" height="30" rx="4" fill="#f1f5f9" stroke="#e2e8f0"/>
  <text x="515" y="50" text-anchor="middle" font-size="9" fill="#475569">Server :50051</text>
  <rect x="470" y="65" width="90" height="30" rx="4" fill="#f1f5f9" stroke="#e2e8f0"/>
  <text x="515" y="85" text-anchor="middle" font-size="9" fill="#475569">Server :50052</text>

  <!-- Channel states -->
  <text x="10" y="145" font-size="8" fill="#64748b">Channel states: </text>
  <rect x="85" y="135" width="60" height="14" rx="3" fill="#dcfce7"/>
  <text x="115" y="145" text-anchor="middle" font-size="8" fill="#15803d">READY</text>
  <rect x="150" y="135" width="75" height="14" rx="3" fill="#e0f2fe"/>
  <text x="187" y="145" text-anchor="middle" font-size="8" fill="#0369a1">CONNECTING</text>
  <rect x="230" y="135" width="50" height="14" rx="3" fill="#fef3c7"/>
  <text x="255" y="145" text-anchor="middle" font-size="8" fill="#92400e">IDLE</text>
  <rect x="285" y="135" width="95" height="14" rx="3" fill="#fef2f2"/>
  <text x="332" y="145" text-anchor="middle" font-size="8" fill="#dc2626">TRANSIENT FAIL</text>
  <rect x="385" y="135" width="65" height="14" rx="3" fill="#f1f5f9"/>
  <text x="417" y="145" text-anchor="middle" font-size="8" fill="#64748b">SHUTDOWN</text>
</svg>
</div>
```

  - Code block: `grpc.Dial` / `grpc.NewClient` (Go) với WithTransportCredentials, WithKeepalive, WithDefaultServiceConfig
  - Sub-section "KeepAlive config": GRPC_KEEPALIVE_TIME_MS, GRPC_KEEPALIVE_TIMEOUT_MS, permit without streams
  - Sub-section "Stub types (Go)": `NewXxxClient` (non-blocking), `NewXxxClient` with `grpc.WaitForReady`

- [ ] **s11 — Metadata**
  - Code block: sending metadata (Go) — `metadata.New(map[string]string{...})`, `metadata.AppendToOutgoingContext`
  - Code block: receiving metadata (Go) — `header, _ := metadata.FromIncomingContext(ctx)`
  - Sub-section "Binary metadata": key ending in `-bin`, base64 encoded automatically
  - Sub-section "Trailing metadata": sent after response, use `grpc.SetTrailer`
  - Box blue "Common patterns": auth token in metadata, distributed tracing (trace-id/span-id), rate limiting headers

---

## Task 16 — `grpc-deep-dive.html`: Sections s12–s13

- [ ] **s12 — Error Handling**
  - Table: Status Code | gRPC Code | HTTP Equiv | When to use — all 16 codes: OK/CANCELLED/UNKNOWN/INVALID_ARGUMENT/DEADLINE_EXCEEDED/NOT_FOUND/ALREADY_EXISTS/PERMISSION_DENIED/RESOURCE_EXHAUSTED/FAILED_PRECONDITION/ABORTED/OUT_OF_RANGE/UNIMPLEMENTED/INTERNAL/UNAVAILABLE/DATA_LOSS/UNAUTHENTICATED
  - Code block: returning status with details (Go) — `status.Errorf(codes.NotFound, "user %d not found", id)`
  - Sub-section "Rich Error Details": `google.rpc.ErrorInfo`, `google.rpc.RetryInfo`, `google.rpc.BadRequest`, `google.rpc.QuotaFailure`
  - Code block: attaching details to status (Go)
  - Box orange "UNKNOWN vs INTERNAL": UNKNOWN = status không map được (cross-language), INTERNAL = unexpected server error

- [ ] **s13 — Deadlines & Cancellation**
  - Code block: context with deadline (Go) — `ctx, cancel := context.WithTimeout(ctx, 5*time.Second)`
  - Sub-section "Deadline Propagation": client deadline tự động propagate qua toàn call chain; server đọc via `ctx.Deadline()`
  - Sub-section "Graceful vs Abrupt": `ctx.Done()` → CANCELLED; deadline exceeded → DEADLINE_EXCEEDED
  - Sub-section "Retry vs Hedging": retry = gửi lại sau fail; hedging = gửi song song N requests, lấy cái về trước
  - Box blue "Best practice": luôn set deadline cho production RPC; server phải check `ctx.Err()` trong long-running operations

---

## Task 17 — `grpc-deep-dive.html`: Section s14 + SVG 5 (Interceptor Chain)

- [ ] **s14 — Interceptors** + SVG 5

  SVG 5 — Interceptor chain (viewBox="0 0 700 100"):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr-i" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#0ea5e9"/>
    </marker>
    <marker id="arr-i-ret" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#22c55e"/>
    </marker>
  </defs>
  <!-- Boxes -->
  <rect x="10" y="35" width="80" height="30" rx="4" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="50" y="54" text-anchor="middle" font-size="9" font-weight="700" fill="#0369a1">RPC Call</text>

  <rect x="120" y="35" width="90" height="30" rx="4" fill="#fef3c7" stroke="#fde68a"/>
  <text x="165" y="51" text-anchor="middle" font-size="9" font-weight="700" fill="#92400e">Auth</text>
  <text x="165" y="62" text-anchor="middle" font-size="8" fill="#92400e">interceptor</text>

  <rect x="240" y="35" width="90" height="30" rx="4" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="285" y="51" text-anchor="middle" font-size="9" font-weight="700" fill="#7c3aed">Logging</text>
  <text x="285" y="62" text-anchor="middle" font-size="8" fill="#7c3aed">interceptor</text>

  <rect x="360" y="35" width="90" height="30" rx="4" fill="#fff7ed" stroke="#fdba74"/>
  <text x="405" y="51" text-anchor="middle" font-size="9" font-weight="700" fill="#c2410c">Retry</text>
  <text x="405" y="62" text-anchor="middle" font-size="8" fill="#c2410c">interceptor</text>

  <rect x="480" y="35" width="90" height="30" rx="4" fill="#f0fdf4" stroke="#86efac"/>
  <text x="525" y="51" text-anchor="middle" font-size="9" font-weight="700" fill="#15803d">Metrics</text>
  <text x="525" y="62" text-anchor="middle" font-size="8" fill="#15803d">interceptor</text>

  <rect x="600" y="35" width="90" height="30" rx="4" fill="#dcfce7" stroke="#86efac"/>
  <text x="645" y="54" text-anchor="middle" font-size="9" font-weight="700" fill="#15803d">Handler</text>

  <!-- Forward arrows -->
  <line x1="90" y1="46" x2="118" y2="46" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-i)"/>
  <line x1="210" y1="46" x2="238" y2="46" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-i)"/>
  <line x1="330" y1="46" x2="358" y2="46" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-i)"/>
  <line x1="450" y1="46" x2="478" y2="46" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-i)"/>
  <line x1="570" y1="46" x2="598" y2="46" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arr-i)"/>

  <!-- Return arrows -->
  <line x1="118" y1="58" x2="90" y2="58" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-i-ret)"/>
  <line x1="238" y1="58" x2="210" y2="58" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-i-ret)"/>
  <line x1="358" y1="58" x2="330" y2="58" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-i-ret)"/>
  <line x1="478" y1="58" x2="450" y2="58" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-i-ret)"/>
  <line x1="598" y1="58" x2="570" y2="58" stroke="#22c55e" stroke-width="1.5" marker-end="url(#arr-i-ret)"/>

  <text x="350" y="15" text-anchor="middle" font-size="9" fill="#94a3b8">→ request flow (blue)   ← response flow (green)</text>
</svg>
</div>
```

  - Code block: unary interceptor signature (Go):
    ```go
    func AuthInterceptor(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
        if err := validateToken(ctx); err != nil { return nil, status.Errorf(codes.Unauthenticated, "invalid token") }
        return handler(ctx, req)
    }
    ```
  - Code block: chaining interceptors — `grpc.ChainUnaryInterceptor(auth, logging, retry, metrics)`
  - Sub-section "Stream interceptors": `grpc.StreamServerInterceptor`, wraps `grpc.ServerStream`
  - Box blue "Common interceptors": grpc-ecosystem/go-grpc-middleware library — auth, recovery, validator, zap, prometheus

---

## Task 18 — `grpc-deep-dive.html`: Section s15 + SVG 6 (Load Balancing)

- [ ] **s15 — Load Balancing** + SVG 6

  SVG 6 — LB strategies (viewBox="0 0 700 180"):

```html
<div class="diagram-wrap">
<svg class="diagram" viewBox="0 0 700 180" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr-lb" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#64748b"/>
    </marker>
  </defs>

  <!-- Left: Client-side LB -->
  <text x="160" y="14" text-anchor="middle" font-size="11" font-weight="700" fill="#0369a1">Client-Side Load Balancing</text>
  <rect x="30" y="25" width="80" height="30" rx="4" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="70" y="45" text-anchor="middle" font-size="9" font-weight="700" fill="#0369a1">Client</text>
  <text x="70" y="65" text-anchor="middle" font-size="8" fill="#64748b">(round_robin)</text>

  <line x1="110" y1="35" x2="148" y2="30" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-lb)"/>
  <line x1="110" y1="40" x2="148" y2="55" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-lb)"/>
  <line x1="110" y1="45" x2="148" y2="80" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-lb)"/>

  <rect x="150" y="18" width="90" height="22" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="195" y="33" text-anchor="middle" font-size="9" fill="#15803d">Server A :50051</text>
  <rect x="150" y="44" width="90" height="22" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="195" y="59" text-anchor="middle" font-size="9" fill="#15803d">Server B :50052</text>
  <rect x="150" y="70" width="90" height="22" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="195" y="85" text-anchor="middle" font-size="9" fill="#15803d">Server C :50053</text>

  <text x="160" y="110" text-anchor="middle" font-size="8" fill="#64748b">✓ Low latency (direct connect)</text>
  <text x="160" y="122" text-anchor="middle" font-size="8" fill="#64748b">✗ Client must know all servers</text>

  <!-- Right: Proxy LB (Envoy) -->
  <text x="530" y="14" text-anchor="middle" font-size="11" font-weight="700" fill="#7c3aed">Proxy Load Balancing</text>
  <rect x="380" y="55" width="80" height="30" rx="4" fill="#e0f2fe" stroke="#7dd3fc"/>
  <text x="420" y="75" text-anchor="middle" font-size="9" font-weight="700" fill="#0369a1">Client</text>

  <line x1="460" y1="70" x2="498" y2="70" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-lb)"/>

  <rect x="500" y="45" width="80" height="50" rx="4" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="540" y="65" text-anchor="middle" font-size="9" font-weight="700" fill="#7c3aed">Envoy</text>
  <text x="540" y="78" text-anchor="middle" font-size="8" fill="#7c3aed">Proxy / Sidecar</text>
  <text x="540" y="89" text-anchor="middle" font-size="8" fill="#7c3aed">xDS control plane</text>

  <line x1="580" y1="60" x2="618" y2="45" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-lb)"/>
  <line x1="580" y1="70" x2="618" y2="70" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-lb)"/>
  <line x1="580" y1="80" x2="618" y2="95" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-lb)"/>

  <rect x="620" y="33" width="70" height="22" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="655" y="48" text-anchor="middle" font-size="9" fill="#15803d">Server A</text>
  <rect x="620" y="59" width="70" height="22" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="655" y="74" text-anchor="middle" font-size="9" fill="#15803d">Server B</text>
  <rect x="620" y="85" width="70" height="22" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="655" y="100" text-anchor="middle" font-size="9" fill="#15803d">Server C</text>

  <text x="530" y="128" text-anchor="middle" font-size="8" fill="#64748b">✓ Client decoupled from topology</text>
  <text x="530" y="140" text-anchor="middle" font-size="8" fill="#64748b">✗ Extra hop latency</text>

  <!-- Divider -->
  <line x1="350" y1="10" x2="350" y2="155" stroke="#e2e8f0" stroke-dasharray="4 2"/>
</svg>
</div>
```

  - Table: LB policy | Behavior | When to use — pick_first/round_robin/least_conn/xDS
  - Sub-section "xDS Protocol": Envoy data plane API, ADS (Aggregated Discovery Service), Istio as control plane
  - Sub-section "Service Mesh": sidecar pattern, mTLS between services, observability (Jaeger/Zipkin), traffic management

---

## Task 19 — `grpc-deep-dive.html`: Section s16 (gRPC-Web + Production)

- [ ] **s16 — gRPC-Web, Transcoding & Production**
  - Sub-section "gRPC-Web": browsers không support HTTP/2 trailers → grpc-web protocol, Envoy proxy translate, `grpc-web` vs `grpc-web-text` (base64), `@grpc/grpc-js` không hoạt động in browser
  - Code block: Envoy config snippet cho grpc-web transcoding
  - Sub-section "gRPC Gateway (Transcoding)": `google.api.http` annotation trong .proto, protoc-gen-grpc-gateway tạo REST reverse proxy, OpenAPI spec generation
  - Code block: proto annotation example:
    ```protobuf
    rpc GetUser(GetUserRequest) returns (User) {
      option (google.api.http) = { get: "/v1/users/{user_id}" };
    }
    ```
  - Sub-section "mTLS": channel credentials với client cert, `grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{...}))`
  - Sub-section "Health Checking": `grpc.health.v1.Health` service (Check + Watch), `grpc-health-probe` CLI tool, Kubernetes liveness/readiness probe integration
  - Sub-section "Server Reflection": `grpc.reflection.v1alpha`, enables `grpcurl` CLI, dynamic client discovery
  - **Performance comparison table**: gRPC vs REST/JSON — payload size (binary ~3-10x smaller), serialization speed, streaming support, browser support, code gen, human readability

- [ ] Final verification: open `grpc-deep-dive.html`, scroll all 16 sections, sidebar active state, all 6 SVGs render

---

## Self-Review Checklist

- [ ] All 18 sections of `http-deep-dive.html` implemented with content
- [ ] All 7 SVGs in `http-deep-dive.html` present and rendering
- [ ] All 16 sections of `grpc-deep-dive.html` implemented with content
- [ ] All 6 SVGs in `grpc-deep-dive.html` present and rendering
- [ ] IntersectionObserver active state working in both files
- [ ] No horizontal overflow (all SVGs have `overflow: hidden` or fit in viewBox)
- [ ] Code blocks use correct `<pre>` + HTML entity escaping (`&lt;`, `&gt;`, `&amp;`)
- [ ] All Vietnamese text correct (no English placeholders)
