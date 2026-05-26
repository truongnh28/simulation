// Protocol Lab Server — echo + per-connection metrics
// Supports: text frames, binary frames, compressed frames (permessage-deflate)
//
// Client → Server messages:
//   JSON text:  { type: 'echo',  payload: '...',  sentAt: <ms> }
//   JSON text:  { type: 'bench', count: N, size: N }  — server sends N messages of N bytes back
//   JSON text:  { type: 'ping-lat', id: N, sentAt: <ms> }  — explicit latency ping
//   Binary:     ArrayBuffer with arbitrary bytes (echoed back as-is)
//
// Server → Client messages:
//   JSON text:  { type: 'echo',    payload: '...',   echoedAt: <ms>, sentAt: <ms>, bytes: N }
//   JSON text:  { type: 'bench-chunk', data: '...', seq: N }
//   JSON text:  { type: 'bench-done', total: N }
//   JSON text:  { type: 'pong-lat', id: N, sentAt: <ms>, echoedAt: <ms> }
//   JSON text:  { type: 'stats',   ...metrics }  — sent every 1s
//   Binary:     same ArrayBuffer echoed back

'use strict';

const http = require('http');
const { WebSocketServer, WebSocket } = require('ws');
const zlib = require('zlib');

const PORT = process.env.PORT || 3002;

const server = http.createServer((req, res) => {
  const fs = require('fs');
  const path = require('path');
  if (req.url === '/' || req.url === '/index.html') {
    const file = path.join(__dirname, 'index.html');
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(data);
    });
  } else {
    res.writeHead(404); res.end();
  }
});

// Enable permessage-deflate (per-message compression)
const wss = new WebSocketServer({
  server,
  maxPayload: 50 * 1024 * 1024,
  perMessageDeflate: {
    zlibDeflateOptions: { level: zlib.constants.Z_DEFAULT_COMPRESSION },
    zlibInflateOptions: { chunkSize: 10 * 1024 },
    serverNoContextTakeover: false,
    clientNoContextTakeover: false,
    threshold: 0, // compress even small messages when client requests it
  },
});

wss.on('connection', (ws, req) => {
  const connectedAt = Date.now();
  ws.isAlive = true;
  ws.on('pong', () => { ws.isAlive = true; });

  // Per-connection metrics
  const metrics = {
    msgCount: 0,
    bytesReceived: 0,
    bytesSent: 0,
    latencySamples: [],   // RTT samples (ms)
    msgRateSamples: [],   // { ts, count } — for msgs/s calculation
    byteRateSamples: [],  // { ts, bytes } — for KB/s calculation
    textFrames: 0,
    binaryFrames: 0,
    lastWindow: { count: 0, bytes: 0, ts: Date.now() },
  };

  // Send stats every second
  const statsTimer = setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN) return;
    const now = Date.now();
    const windowMs = now - metrics.lastWindow.ts;
    const msgRate = windowMs > 0 ? (metrics.lastWindow.count / (windowMs / 1000)) : 0;
    const byteRate = windowMs > 0 ? (metrics.lastWindow.bytes / (windowMs / 1000)) : 0;

    // Keep last 60 samples for charting
    metrics.msgRateSamples.push({ ts: now, value: msgRate });
    metrics.byteRateSamples.push({ ts: now, value: byteRate });
    if (metrics.msgRateSamples.length > 60) metrics.msgRateSamples.shift();
    if (metrics.byteRateSamples.length > 60) metrics.byteRateSamples.shift();

    const avgLat = metrics.latencySamples.length
      ? metrics.latencySamples.reduce((s, v) => s + v, 0) / metrics.latencySamples.length
      : 0;
    const p50 = percentile(metrics.latencySamples, 50);
    const p95 = percentile(metrics.latencySamples, 95);
    const p99 = percentile(metrics.latencySamples, 99);

    ws.send(JSON.stringify({
      type: 'stats',
      connectedAt,
      uptimeMs: now - connectedAt,
      msgCount: metrics.msgCount,
      bytesReceived: metrics.bytesReceived,
      bytesSent: metrics.bytesSent,
      textFrames: metrics.textFrames,
      binaryFrames: metrics.binaryFrames,
      msgRate: +msgRate.toFixed(2),
      byteRate: +byteRate.toFixed(0),
      msgRateSamples: metrics.msgRateSamples,
      byteRateSamples: metrics.byteRateSamples,
      latSamples: metrics.latencySamples.slice(-60),
      latAvg: +avgLat.toFixed(2),
      latP50: p50,
      latP95: p95,
      latP99: p99,
      sampleCount: metrics.latencySamples.length,
    }));

    // Reset window
    metrics.lastWindow = { count: 0, bytes: 0, ts: now };
  }, 1000);

  ws.on('message', (data, isBinary) => {
    const bytes = isBinary ? data.byteLength : Buffer.byteLength(data);
    metrics.msgCount++;
    metrics.bytesReceived += bytes;
    metrics.lastWindow.count++;
    metrics.lastWindow.bytes += bytes;

    if (isBinary) {
      metrics.binaryFrames++;
      // Echo binary back immediately
      ws.send(data);
      metrics.bytesSent += data.byteLength;
      return;
    }

    metrics.textFrames++;

    let msg;
    try { msg = JSON.parse(data); } catch {
      // Echo raw text back
      ws.send(data);
      metrics.bytesSent += Buffer.byteLength(data);
      return;
    }

    switch (msg.type) {
      case 'echo': {
        const resp = JSON.stringify({
          type: 'echo',
          payload: msg.payload,
          sentAt: msg.sentAt,
          echoedAt: Date.now(),
          bytes,
        });
        ws.send(resp);
        metrics.bytesSent += Buffer.byteLength(resp);
        if (msg.sentAt) {
          const rtt = Date.now() - msg.sentAt;
          metrics.latencySamples.push(rtt);
          if (metrics.latencySamples.length > 1000) metrics.latencySamples.shift();
        }
        break;
      }

      case 'ping-lat': {
        const resp = JSON.stringify({ type: 'pong-lat', id: msg.id, sentAt: msg.sentAt, echoedAt: Date.now() });
        ws.send(resp);
        metrics.bytesSent += Buffer.byteLength(resp);
        break;
      }

      case 'bench': {
        // Server sends `count` messages of `size` bytes back to client
        const count = Math.min(Number(msg.count) || 100, 10000);
        const size = Math.min(Number(msg.size) || 128, 65536);
        const payload = 'A'.repeat(size);
        let sent = 0;

        function sendNext() {
          if (ws.readyState !== WebSocket.OPEN) return;
          while (sent < count) {
            const resp = JSON.stringify({ type: 'bench-chunk', data: payload, seq: sent });
            ws.send(resp);
            metrics.bytesSent += Buffer.byteLength(resp);
            sent++;
            // Yield every 50 messages to not block event loop
            if (sent % 50 === 0 && ws.bufferedAmount > 65536) {
              setImmediate(sendNext);
              return;
            }
          }
          const done = JSON.stringify({ type: 'bench-done', total: count });
          ws.send(done);
          metrics.bytesSent += Buffer.byteLength(done);
        }
        sendNext();
        break;
      }

      default:
        // Unknown: echo back
        ws.send(data);
        metrics.bytesSent += Buffer.byteLength(data);
    }
  });

  ws.on('close', () => {
    clearInterval(statsTimer);
  });

  ws.on('error', err => {
    console.error('ws error:', err.message);
    clearInterval(statsTimer);
  });
});

// ─── Heartbeat ────────────────────────────────────────────────────────────────
const hb = setInterval(() => {
  wss.clients.forEach(ws => {
    if (!ws.isAlive) { ws.terminate(); return; }
    ws.isAlive = false;
    ws.ping();
  });
}, 30_000);

wss.on('close', () => clearInterval(hb));

// ─── Helpers ──────────────────────────────────────────────────────────────────
function percentile(arr, p) {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

// ─── Start ────────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`Protocol Lab server listening on http://localhost:${PORT}`);
  console.log('WebSocket endpoint: ws://localhost:' + PORT);
  console.log('permessage-deflate: enabled');
});

process.on('SIGTERM', () => {
  wss.clients.forEach(ws => ws.close(1001, 'Server shutting down'));
  server.close(() => process.exit(0));
});
