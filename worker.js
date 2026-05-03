// ─────────────────────────────────────────────
// ENV variables (Cloudflare Dashboard):
//   TELEGRAM_BOT_TOKEN
//   NVIDIA_API_KEY
//   OWNER_ID
//   KV  → KV namespace binding named "KV"
// ─────────────────────────────────────────────

const NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions";
const NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models";
const DEFAULT_MODEL = "openai/gpt-oss-20b";
const MAX_CONTEXT = 40;

// ─────────────────────────────────────────────
// KV helpers
// ─────────────────────────────────────────────

async function kvGet(env, key) {
  try {
    const val = await env.KV.get(key);
    return val ? JSON.parse(val) : null;
  } catch {
    return null;
  }
}

async function kvSet(env, key, value, ttl = null) {
  const opts = ttl ? { expirationTtl: ttl } : {};
  await env.KV.put(key, JSON.stringify(value), opts);
}

async function kvDel(env, key) {
  await env.KV.delete(key);
}

// ─────────────────────────────────────────────
// History
// ─────────────────────────────────────────────

async function getHistory(env, chatId) {
  const data = await kvGet(env, `history:${chatId}`);
  return Array.isArray(data) ? data : [];
}

async function saveMsg(env, chatId, role, content) {
  let history = await getHistory(env, chatId);
  history.push({ role, content });
  if (history.length > MAX_CONTEXT) history = history.slice(-MAX_CONTEXT);
  await kvSet(env, `history:${chatId}`, history);
}

async function resetHistory(env, chatId) {
  await kvDel(env, `history:${chatId}`);
}

// ─────────────────────────────────────────────
// Model per chat
// ─────────────────────────────────────────────

async function getModel(env, chatId) {
  return (await kvGet(env, `model:${chatId}`)) || DEFAULT_MODEL;
}

async function saveModel(env, chatId, model) {
  await kvSet(env, `model:${chatId}`, model);
}

// ─────────────────────────────────────────────
// Access control
// ─────────────────────────────────────────────

function isOwner(env, userId) {
  return String(userId) === String(env.OWNER_ID);
}

async function isApproved(env, userId) {
  if (isOwner(env, userId)) return true;
  return (await kvGet(env, `approved:${userId}`)) === true;
}

async function approveUser(env, userId) {
  await kvSet(env, `approved:${userId}`, true);
}

async function unapproveUser(env, userId) {
  await kvDel(env, `approved:${userId}`);
}

// ─────────────────────────────────────────────
// Auto model params
// 1. Fetch max_context_length from NVIDIA /v1/models/{model} (cached 24h in KV)
// 2. Detect thinking model by name prefix
// 3. Set temperature/top_p accordingly
// ─────────────────────────────────────────────

const THINKING_PREFIXES = [
  "moonshotai/kimi",
  "deepseek-ai/deepseek-r1",
  "qwen/qwen3",
  "nvidia/llama-3.1-nemotron-ultra",
];

function isThinkingModel(model) {
  const m = model.toLowerCase();
  return THINKING_PREFIXES.some((p) => m.startsWith(p));
}

async function fetchModelMaxTokens(env, model) {
  const cacheKey = `modelinfo:${model}`;
  const cached = await kvGet(env, cacheKey);
  if (cached !== null) return cached;

  try {
    const resp = await fetch(`${NVIDIA_MODELS_URL}/${encodeURIComponent(model)}`, {
      headers: { Authorization: `Bearer ${env.NVIDIA_API_KEY}` },
    });
    if (resp.ok) {
      const data = await resp.json();
      const ctxLen = data?.max_context_length || data?.context_window || null;
      if (ctxLen) {
        const maxOut = Math.min(Math.floor(ctxLen / 2), 32768);
        await kvSet(env, cacheKey, maxOut, 86400);
        return maxOut;
      }
    }
  } catch {
    // ignore, return null
  }
  return null;
}

async function getModelParams(env, model) {
  const thinking = isThinkingModel(model);
  const fetched = await fetchModelMaxTokens(env, model);
  const maxTokens = fetched || (thinking ? 16384 : 4096);

  const params = {
    max_tokens: maxTokens,
    temperature: thinking ? 1.0 : 0.7,
    top_p: thinking ? 1.0 : 0.9,
    stream: false,
  };

  if (thinking) params.chat_template_kwargs = { thinking: true };

  return params;
}

// ─────────────────────────────────────────────
// NVIDIA API call
// ─────────────────────────────────────────────

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function callNvidiaAPI(env, messages, model) {
  const retryWaits = [3000, 6000, 12000, 20000];
  const params = await getModelParams(env, model);

  for (let attempt = 0; attempt < retryWaits.length; attempt++) {
    let resp;
    try {
      resp = await fetch(NVIDIA_API_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.NVIDIA_API_KEY}`,
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ model, messages, ...params }),
      });
    } catch {
      if (attempt < retryWaits.length - 1) { await sleep(retryWaits[attempt]); continue; }
      return "API timeout. Thoda baad try karo.";
    }

    if (resp.ok) {
      const data = await resp.json();
      return data?.choices?.[0]?.message?.content?.trim() || "Empty response.";
    }

    if (resp.status === 429) {
      if (attempt < retryWaits.length - 1) { await sleep(retryWaits[attempt]); continue; }
      return "Rate limit. Thoda ruk ke dobara try karo.";
    }

    if ([500, 502, 503, 504].includes(resp.status)) {
      if (attempt < retryWaits.length - 1) { await sleep(retryWaits[attempt]); continue; }
      return `Server busy (${resp.status}). Baad mein try karo.`;
    }

    const errText = await resp.text();
    return `API Error ${resp.status}: ${errText.slice(0, 500)}`;
  }

  return "API busy. Thoda baad try karo.";
}

// ─────────────────────────────────────────────
// Text formatting
// ─────────────────────────────────────────────

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function applyInline(text) {
  text = escapeHtml(text);
  text = text.replace(/\*\*([^\n*][\s\S]*?[^\n*])\*\*/g, (_, m) => `<b>${m}</b>`);
  text = text.replace(/`([^`\n]+)`/g, (_, m) => `<code>${m}</code>`);
  return text;
}

function formatResponse(text) {
  if (!text) return "";
  text = text.replace(/\r\n/g, "\n").replace(/###/g, "").replace(/\n{5,}/g, "\n\n\n");

  const parts = [];
  let pos = 0;
  const pattern = /```(?:[a-zA-Z0-9_+\-.]*)\n?([\s\S]*?)```/g;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    const before = text.slice(pos, match.index);
    if (before) parts.push(applyInline(before));
    const code = escapeHtml(match[1].replace(/^\n/, "").replace(/\n$/, ""));
    parts.push(`<pre>${code}</pre>`);
    pos = match.index + match[0].length;
  }

  const rest = text.slice(pos);
  if (rest) parts.push(applyInline(rest));

  let out = parts.join("").replace(/\n{5,}/g, "\n\n\n");
  const count = (text.match(/```/g) || []).length;
  if (count % 2 !== 0) out += "\n<code>...</code>";

  return out.trim();
}

function splitText(text, limit = 3900) {
  if (!text || text.length <= limit) return [text || ""];
  const chunks = [];
  let current = "";
  for (const line of text.split("\n")) {
    if (current.length + line.length + 1 <= limit) {
      current += line + "\n";
    } else {
      if (current.trim()) chunks.push(current.trim());
      current = line + "\n";
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks.length ? chunks : [text.slice(0, limit)];
}

// ─────────────────────────────────────────────
// Telegram helpers
// ─────────────────────────────────────────────

async function tgCall(env, method, body) {
  const url = `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${method}`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return resp.json();
}

async function sendMessage(env, chatId, text, replyTo = null) {
  const html = formatResponse(text);
  const body = {
    chat_id: chatId,
    text: html || text,
    parse_mode: "HTML",
    disable_web_page_preview: true,
  };
  if (replyTo) body.reply_to_message_id = replyTo;

  const result = await tgCall(env, "sendMessage", body);
  if (!result.ok) {
    const fallback = { chat_id: chatId, text: text.slice(0, 4096), disable_web_page_preview: true };
    if (replyTo) fallback.reply_to_message_id = replyTo;
    return tgCall(env, "sendMessage", fallback);
  }
  return result;
}

async function editMessage(env, chatId, messageId, text) {
  const html = formatResponse(text);
  const body = {
    chat_id: chatId,
    message_id: messageId,
    text: html || text,
    parse_mode: "HTML",
    disable_web_page_preview: true,
  };
  const result = await tgCall(env, "editMessageText", body);
  if (!result.ok) {
    return tgCall(env, "editMessageText", {
      chat_id: chatId,
      message_id: messageId,
      text: text.slice(0, 4096),
      disable_web_page_preview: true,
    });
  }
  return result;
}

async function sendTyping(env, chatId) {
  await tgCall(env, "sendChatAction", { chat_id: chatId, action: "typing" });
}

// ─────────────────────────────────────────────
// Commands
// ─────────────────────────────────────────────

async function handleStart(env, update) {
  const { id: userId } = update.message.from;
  const chatId = update.message.chat.id;
  const msgId = update.message.message_id;

  if (!(await isApproved(env, userId))) {
    await sendMessage(env, chatId, `Access denied.\nYour ID: <code>${userId}</code>\nAsk owner to approve.`, msgId);
    return;
  }
  await sendMessage(env, chatId, "Hey, how can I help you today?", msgId);
}

async function handleReset(env, update) {
  const { id: userId } = update.message.from;
  const chatId = update.message.chat.id;
  const msgId = update.message.message_id;

  if (!(await isApproved(env, userId))) { await sendMessage(env, chatId, "Access denied.", msgId); return; }
  await resetHistory(env, chatId);
  await sendMessage(env, chatId, "Memory cleared.", msgId);
}

async function handleSetModel(env, update) {
  const { id: userId } = update.message.from;
  const chatId = update.message.chat.id;
  const msgId = update.message.message_id;
  const args = (update.message.text || "").trim().split(/\s+/).slice(1);

  if (!(await isApproved(env, userId))) { await sendMessage(env, chatId, "Access denied.", msgId); return; }

  if (!args.length) {
    const cur = await getModel(env, chatId);
    await sendMessage(env, chatId, `Model: <code>${cur}</code>`, msgId);
    return;
  }

  const model = args.join(" ").trim();
  const loadMsg = await sendMessage(env, chatId, "Fetching model info...", msgId);
  const loadMsgId = loadMsg?.result?.message_id;

  await saveModel(env, chatId, model);
  const params = await getModelParams(env, model);
  const thinking = isThinkingModel(model);

  const lines = [
    `Model: <code>${model}</code>`,
    `max_tokens: ${params.max_tokens}`,
    `temperature: ${params.temperature}`,
    `top_p: ${params.top_p}`,
    thinking ? `thinking: true` : null,
  ].filter(Boolean).join("\n");

  if (loadMsgId) {
    await editMessage(env, chatId, loadMsgId, lines);
  } else {
    await sendMessage(env, chatId, lines, msgId);
  }
}

async function handleApprove(env, update) {
  const { id: userId } = update.message.from;
  const chatId = update.message.chat.id;
  const msgId = update.message.message_id;
  const args = (update.message.text || "").trim().split(/\s+/).slice(1);

  if (!isOwner(env, userId)) { await sendMessage(env, chatId, "Owner only.", msgId); return; }
  if (!args[0] || !/^\d+$/.test(args[0])) { await sendMessage(env, chatId, "Usage: /approve user_id", msgId); return; }
  await approveUser(env, args[0]);
  await sendMessage(env, chatId, `Approved: <code>${args[0]}</code>`, msgId);
}

async function handleUnapprove(env, update) {
  const { id: userId } = update.message.from;
  const chatId = update.message.chat.id;
  const msgId = update.message.message_id;
  const args = (update.message.text || "").trim().split(/\s+/).slice(1);

  if (!isOwner(env, userId)) { await sendMessage(env, chatId, "Owner only.", msgId); return; }
  if (!args[0] || !/^\d+$/.test(args[0])) { await sendMessage(env, chatId, "Usage: /unapprove user_id", msgId); return; }
  if (args[0] === String(env.OWNER_ID)) { await sendMessage(env, chatId, "Owner ko unapprove nahi kar sakte.", msgId); return; }
  await unapproveUser(env, args[0]);
  await sendMessage(env, chatId, `Unapproved: <code>${args[0]}</code>`, msgId);
}

// ─────────────────────────────────────────────
// Main message handler
// ─────────────────────────────────────────────

async function handleMessage(env, update) {
  const { id: userId } = update.message.from;
  const chatId = update.message.chat.id;
  const msgId = update.message.message_id;
  const text = (update.message.text || "").trim();

  if (!text) return;

  if (!(await isApproved(env, userId))) {
    await sendMessage(env, chatId, `Access denied.\nYour ID: <code>${userId}</code>`, msgId);
    return;
  }

  const [sentResult] = await Promise.all([
    sendMessage(env, chatId, "...", msgId),
    sendTyping(env, chatId),
  ]);
  const sentMsgId = sentResult?.result?.message_id;

  const model = await getModel(env, chatId);
  const history = await getHistory(env, chatId);

  const messages = [
    ...history.map(({ role, content }) => ({ role, content })),
    { role: "user", content: text },
  ];

  const response = await callNvidiaAPI(env, messages, model);
  const chunks = splitText(response, 3900);

  if (sentMsgId) {
    await editMessage(env, chatId, sentMsgId, chunks[0]);
  } else {
    await sendMessage(env, chatId, chunks[0]);
  }

  for (let i = 1; i < chunks.length; i++) {
    await sendMessage(env, chatId, chunks[i]);
  }

  await saveMsg(env, chatId, "user", text);
  await saveMsg(env, chatId, "assistant", response);
}

// ─────────────────────────────────────────────
// Router
// ─────────────────────────────────────────────

async function handleUpdate(env, update) {
  if (!update.message?.text) return;

  const text = (update.message.text || "").trim();
  const cmd = text.split(/\s+/)[0].toLowerCase().replace(/@.*$/, "");

  if (cmd === "/start")     return handleStart(env, update);
  if (cmd === "/reset")     return handleReset(env, update);
  if (cmd === "/setmodel")  return handleSetModel(env, update);
  if (cmd === "/approve")   return handleApprove(env, update);
  if (cmd === "/unapprove") return handleUnapprove(env, update);

  if (!text.startsWith("/")) return handleMessage(env, update);
}

// ─────────────────────────────────────────────
// Cloudflare Worker entry
// ─────────────────────────────────────────────

export default {
  async fetch(request, env) {
    if (request.method === "GET")  return new Response("OK", { status: 200 });
    if (request.method !== "POST") return new Response("Method Not Allowed", { status: 405 });

    let update;
    try {
      update = await request.json();
    } catch {
      return new Response("Bad Request", { status: 400 });
    }

    try {
      await handleUpdate(env, update);
    } catch (e) {
      console.error("handleUpdate error:", e);
    }

    return new Response("OK", { status: 200 });
  },
};
