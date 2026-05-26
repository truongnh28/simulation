// Chat Room Server — Node.js + ws@8 (no Express)
// Protocol: JSON messages for control, ArrayBuffer chunks for file transfer
//
// Message types (client → server):
//   join          { type, name, room }
//   message       { type, text }
//   private       { type, to, text }          to = recipient id
//   typing        { type, isTyping }
//   file-start    { type, name, size, mime, to? }  to = DM target (optional)
//   file-chunk    ArrayBuffer (binary frame)
//   file-end      { type }
//
// Message types (server → client):
//   welcome       { type, id, room, users }
//   message       { type, id, from, text, ts }
//   private       { type, from, text, ts }
//   history       { type, messages: [...] }
//   presence      { type, id, name, online }
//   typing        { type, id, name, isTyping }
//   file-start    { type, id, from, name, size, mime, transferId }
//   file-chunk    ArrayBuffer prefixed with 4-byte transferId (little-endian uint32)
//   file-end      { type, transferId }
//   error         { type, message }

'use strict';

const http = require('http');
const { WebSocketServer, WebSocket } = require('ws');

const PORT = process.env.PORT || 3001;
const HISTORY_MAX = 50;
const MAX_PAYLOAD = 20 * 1024 * 1024; // 20 MB

// ─── State ───────────────────────────────────────────────────────────────────

/** @type {Map<WebSocket, {id: string, name: string, room: string}>} */
const clients = new Map();

/** @type {Map<string, Set<WebSocket>>} */
const rooms = new Map();

/** @type {Map<string, Array>} */
const history = new Map();

/** @type {Map<WebSocket, {id: string, name: string, size: number, mime: string, to?: string, chunks: Buffer[]}>} */
const pendingFiles = new Map();

let nextId = 1;
function genId() { return String(nextId++); }

// ─── Helpers ─────────────────────────────────────────────────────────────────

function send(ws, obj) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function sendBinary(ws, buf) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(buf);
  }
}

/** Broadcast JSON to every member of a room, optionally excluding one client. */
function broadcast(room, obj, exclude = null) {
  const members = rooms.get(room);
  if (!members) return;
  const msg = JSON.stringify(obj);
  for (const ws of members) {
    if (ws !== exclude && ws.readyState === WebSocket.OPEN) {
      ws.send(msg);
    }
  }
}

/** Broadcast binary to every member of a room, optionally excluding one client. */
function broadcastBinary(room, buf, exclude = null) {
  const members = rooms.get(room);
  if (!members) return;
  for (const ws of members) {
    if (ws !== exclude && ws.readyState === WebSocket.OPEN) {
      ws.send(buf);
    }
  }
}

function roomUsers(room) {
  const members = rooms.get(room) || new Set();
  return [...members]
    .map(ws => clients.get(ws))
    .filter(Boolean)
    .map(({ id, name }) => ({ id, name }));
}

function pushHistory(room, msg) {
  if (!history.has(room)) history.set(room, []);
  const arr = history.get(room);
  arr.push(msg);
  if (arr.length > HISTORY_MAX) arr.shift();
}

// ─── HTTP Server (serves the client HTML on /) ────────────────────────────────

const server = http.createServer((req, res) => {
  const fs = require('fs');
  const path = require('path');
  if (req.url === '/' || req.url === '/index.html') {
    const file = path.join(__dirname, 'index.html');
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404); res.end('Not found'); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(data);
    });
  } else {
    res.writeHead(404);
    res.end();
  }
});

// ─── WebSocket Server ─────────────────────────────────────────────────────────

const wss = new WebSocketServer({ server, maxPayload: MAX_PAYLOAD });

wss.on('connection', (ws, req) => {
  // Not joined yet; wait for 'join' message
  ws.isAlive = true;
  ws.on('pong', () => { ws.isAlive = true; });

  ws.on('message', (data, isBinary) => {
    // ── Binary frame: file chunk ──────────────────────────────────────────────
    if (isBinary) {
      const pf = pendingFiles.get(ws);
      if (!pf) return;
      pf.chunks.push(Buffer.from(data));

      const info = clients.get(ws);
      if (!info) return;

      // Prepend 4-byte transferId to binary frame so receiver can route it
      const header = Buffer.allocUnsafe(4);
      header.writeUInt32LE(pf.transferId, 0);
      const out = Buffer.concat([header, Buffer.from(data)]);

      if (pf.to) {
        // DM file — send only to target
        const target = [...clients.entries()].find(([, v]) => v.id === pf.to)?.[0];
        if (target) sendBinary(target, out);
      } else {
        // Room file — broadcast to room
        broadcastBinary(info.room, out, ws);
      }
      return;
    }

    // ── Text frame: JSON message ──────────────────────────────────────────────
    let msg;
    try { msg = JSON.parse(data); } catch { return; }

    const info = clients.get(ws);

    switch (msg.type) {
      // ── Join ───────────────────────────────────────────────────────────────
      case 'join': {
        if (info) {
          // Leave old room
          leaveRoom(ws, info.room);
        }

        const name = String(msg.name || 'Anonymous').slice(0, 32);
        const room = String(msg.room || 'general').slice(0, 32).replace(/[^a-zA-Z0-9-_]/g, '');
        const id = info ? info.id : genId();

        clients.set(ws, { id, name, room });

        if (!rooms.has(room)) rooms.set(room, new Set());
        rooms.get(room).add(ws);

        // Send history
        const hist = history.get(room) || [];
        send(ws, { type: 'history', messages: hist });

        // Welcome
        send(ws, { type: 'welcome', id, room, users: roomUsers(room) });

        // Announce presence
        broadcast(room, { type: 'presence', id, name, online: true }, ws);

        const joinMsg = { type: 'message', id: '0', from: '(server)', text: `${name} joined #${room}`, ts: Date.now(), system: true };
        pushHistory(room, joinMsg);
        broadcast(room, joinMsg);
        break;
      }

      // ── Message ────────────────────────────────────────────────────────────
      case 'message': {
        if (!info) return send(ws, { type: 'error', message: 'Not joined' });
        const text = String(msg.text || '').slice(0, 4000);
        if (!text) return;
        const out = { type: 'message', id: genId(), from: info.name, senderId: info.id, text, ts: Date.now() };
        pushHistory(info.room, out);
        broadcast(info.room, out);
        send(ws, out); // echo back to sender too
        break;
      }

      // ── Private message ────────────────────────────────────────────────────
      case 'private': {
        if (!info) return;
        const text = String(msg.text || '').slice(0, 4000);
        const toId = String(msg.to || '');
        if (!text || !toId) return;
        const target = [...clients.entries()].find(([, v]) => v.id === toId)?.[0];
        if (!target) return send(ws, { type: 'error', message: 'User not found' });
        const dm = { type: 'private', from: info.name, fromId: info.id, text, ts: Date.now() };
        send(target, dm);
        send(ws, { ...dm, to: toId }); // echo to sender
        break;
      }

      // ── Typing indicator ───────────────────────────────────────────────────
      case 'typing': {
        if (!info) return;
        broadcast(info.room, { type: 'typing', id: info.id, name: info.name, isTyping: !!msg.isTyping }, ws);
        break;
      }

      // ── File start ─────────────────────────────────────────────────────────
      case 'file-start': {
        if (!info) return;
        const transferId = Date.now() & 0xFFFFFFFF; // 32-bit
        pendingFiles.set(ws, {
          id: genId(),
          name: String(msg.name || 'file').slice(0, 256),
          size: Number(msg.size) || 0,
          mime: String(msg.mime || 'application/octet-stream').slice(0, 128),
          to: msg.to ? String(msg.to) : null,
          chunks: [],
          transferId,
        });
        const pf = pendingFiles.get(ws);
        const announce = { type: 'file-start', id: pf.id, from: info.name, senderId: info.id, name: pf.name, size: pf.size, mime: pf.mime, transferId };
        if (pf.to) {
          const target = [...clients.entries()].find(([, v]) => v.id === pf.to)?.[0];
          if (target) send(target, announce);
        } else {
          broadcast(info.room, announce, ws);
        }
        break;
      }

      // ── File end ────────────────────────────────────────────────────────────
      case 'file-end': {
        if (!info) return;
        const pf = pendingFiles.get(ws);
        if (!pf) return;
        const endMsg = { type: 'file-end', transferId: pf.transferId };
        if (pf.to) {
          const target = [...clients.entries()].find(([, v]) => v.id === pf.to)?.[0];
          if (target) send(target, endMsg);
        } else {
          broadcast(info.room, endMsg, ws);
        }
        pendingFiles.delete(ws);
        break;
      }

      default:
        send(ws, { type: 'error', message: `Unknown type: ${msg.type}` });
    }
  });

  ws.on('close', () => {
    const info = clients.get(ws);
    if (info) {
      leaveRoom(ws, info.room, true);
      clients.delete(ws);
    }
    pendingFiles.delete(ws);
  });

  ws.on('error', (err) => {
    console.error('ws error:', err.message);
  });
});

function leaveRoom(ws, room, disconnect = false) {
  const members = rooms.get(room);
  if (!members) return;
  members.delete(ws);
  if (members.size === 0) rooms.delete(room);

  const info = clients.get(ws);
  if (!info) return;

  broadcast(room, { type: 'presence', id: info.id, name: info.name, online: false });

  if (disconnect) {
    const leaveMsg = { type: 'message', id: '0', from: '(server)', text: `${info.name} left`, ts: Date.now(), system: true };
    pushHistory(room, leaveMsg);
    broadcast(room, leaveMsg);
  }
}

// ─── Heartbeat — detect dead connections ─────────────────────────────────────

const heartbeat = setInterval(() => {
  wss.clients.forEach(ws => {
    if (!ws.isAlive) { ws.terminate(); return; }
    ws.isAlive = false;
    ws.ping();
  });
}, 30_000);

wss.on('close', () => clearInterval(heartbeat));

// ─── Start ────────────────────────────────────────────────────────────────────

server.listen(PORT, () => {
  console.log(`Chat Room server listening on http://localhost:${PORT}`);
  console.log('WebSocket endpoint: ws://localhost:' + PORT);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  wss.clients.forEach(ws => ws.close(1001, 'Server shutting down'));
  server.close(() => process.exit(0));
});
