#!/usr/bin/env python3
"""Telegram bridge for the Semia bot.

Polls the Telegram Bot API, forwards private messages (owner-only) to
`opencode run` (persistent session, JSON events), and sends the model's final
text back to Telegram. One message processed at a time.

Config: .env in this directory.
  TELEGRAM_BOT_TOKEN, SEMA_OWNER_ID, SEMA_MODEL, OPENCODE_BIN,
  optional SEMA_ALLOWED_USERS (comma-sep).
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def load_env():
    env = {}
    with open(os.path.join(HERE, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


ENV = load_env()
TOKEN = ENV.get("TELEGRAM_BOT_TOKEN", "")
OWNER = int(ENV.get("SEMA_OWNER_ID", "0"))
ALLOWED = {int(x) for x in ENV.get("SEMA_ALLOWED_USERS", str(OWNER)).split(",") if x.strip()}
MODEL = ENV.get("SEMA_MODEL", "opencode/big-pickle")
OPENCODE = ENV.get("OPENCODE_BIN", "opencode")
CHANNEL = int(ENV.get("SEMA_CHANNEL_ID", "0") or 0)
CHANNEL_NAME = ENV.get("SEMA_CHANNEL_NAME", "@RationalShelf")
BASE = f"https://api.telegram.org/bot{TOKEN}"
BOT_INFO_CACHE = None
MARKER = "[ПУБЛІКУВАТИ]"
# Publication happens ONLY after an explicit owner confirmation. Generic
# "так"/"ок" is not enough — the owner must explicitly ask to publish.
APPROVE_WORDS = {"так, публікуй", "так публікуй", "опублікуй", "публікуй",
                 "публікуй так", "yes, publish", "publish"}
DENY_WORDS = {"ні", "нет", "no", "n", "скасувати", "не треба", "не публікуй", "ніт",
              "не хочу"}
POST_TRIGGERS = ("пост", "постом", "публікуй", "опублікуй", "канал", "каналу", "каналі")
LEAK_WORDS = ("перевірен", "перевірка", "qdrant", "таблиця", "знайдено", "точний збіг",
              "підтвердж", "traceback", "[score", "=== semantic", "впевненість", "уривок",
              "chunk", "індексован", "зрізана", "обрізана",
              "не публікую", "не публікуємо", "не публікуй в каналі",
              "в особистому діалозі", "особистому", "завданням мостом")


def bot_info():
    global BOT_INFO_CACHE
    if BOT_INFO_CACHE is None:
        BOT_INFO_CACHE = call("getMe", {}).get("result") or {}
    return BOT_INFO_CACHE


def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {}


def save_state(state):
    try:
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


def call(method, params, timeout=30):
    req = urllib.request.Request(BASE + "/" + method,
                                 data=json.dumps(params).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return json.loads(body)
        except Exception:
            return {"ok": False, "description": body.decode("utf-8", "replace")}
    except Exception as e:
        return {"ok": False, "description": str(e)}


def split_text(text, n=4000):
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= n:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) + 2 > n:
            parts.append(cur.strip())
            cur = para
        else:
            cur = (cur + "\n\n" + para).strip()
    if cur:
        parts.append(cur.strip())
    out = []
    for p in parts:
        while len(p) > n:
            out.append(p[:n])
            p = p[n:]
        out.append(p)
    return out


def send_text(chat_id, text, reply_to=None):
    for chunk in split_text(text):
        params = {"chat_id": chat_id, "text": chunk}
        if reply_to:
            params["reply_to_message_id"] = reply_to
        r = call("sendMessage", params)
        if not r.get("ok"):
            print(f"[send fail] {r.get('description')}", file=sys.stderr)
        reply_to = None


def keep_typing(chat_id, stop):
    while not stop.is_set():
        call("sendChatAction", {"chat_id": chat_id, "action": "typing"}, timeout=10)
        stop.wait(4.5)


def run_opencode(message, chat_id):
    state = load_state()
    sid = state.get("session_id")
    cmd = [OPENCODE, "run", "--auto", "--model", MODEL, "--format", "json"]
    if sid:
        cmd += ["--session", sid]
    cmd += ["--message", message]

    stop = threading.Event()
    t = threading.Thread(target=keep_typing, args=(chat_id, stop), daemon=True)
    t.start()
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, timeout=1500,
                              encoding="utf-8", errors="replace",
                              creationflags=CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return "⏳ Запит тривав задовго — скороти або розбий його."
    finally:
        stop.set()

    out = proc.stdout
    err = proc.stderr
    if err:
        print("[stderr]", err[-3000:], file=sys.stderr)

    parts, new_sid = [], None
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("sessionID"):
            new_sid = ev["sessionID"]
        if ev.get("type") == "text" and ev.get("part", {}).get("text"):
            parts.append(ev["part"]["text"])

    if new_sid and new_sid != sid:
        state["session_id"] = new_sid
        save_state(state)

    reply = "\n".join(parts).strip() or "(без відповіді)"
    return reply


MARKER_RE = re.compile(r"^\s*\[ПУБЛІКУВАТИ\]\s*$", re.MULTILINE)


def parse_reply(reply):
    reply = reply or ""
    m = MARKER_RE.search(reply)
    if m:
        return reply[:m.start()].strip(), reply[m.end():].strip()
    return reply.strip(), None


def get_pending_post():
    return load_state().get("pending_post") or ""


def set_pending_post(text):
    st = load_state()
    st["pending_post"] = text
    save_state(st)


def clear_pending_post():
    st = load_state()
    st.pop("pending_post", None)
    save_state(st)


def publish_post(text):
    if not CHANNEL:
        return False
    r = call("sendMessage", {"chat_id": CHANNEL, "text": text, "parse_mode": "HTML"})
    return bool(r.get("ok"))


def handle_message(msg):
    if not msg:
        return
    chat = msg.get("chat", {})
    chat_type = chat.get("type", "")
    chat_id = msg["chat"]["id"]
    sender = msg.get("from", {}).get("id")
    text = msg.get("text")
    if text is None:
        text = msg.get("caption")
    if not text:
        return

    if chat_type == "private":
        is_target = True
    elif chat_type in ("group", "supergroup"):
        self_id = bot_info().get("id")
        self_name = ((bot_info().get("username") or "").lower())
        is_target = False
        entities = msg.get("entities") or msg.get("caption_entities") or []
        for ent in entities:
            if ent.get("type") == "mention":
                frag = text[ent["offset"]:ent["offset"] + ent["length"]]
                if frag.lower() == "@" + self_name:
                    is_target = True
                    cleaned = (text[:ent["offset"]] + text[ent["offset"] + ent["length"]]).strip()
                    if cleaned:
                        text = cleaned
                    break
        if not is_target:
            reply = msg.get("reply_to_message") or {}
            if reply.get("from", {}).get("id") == self_id:
                is_target = True
    else:
        is_target = False

    if not is_target:
        return
    if not text:
        return

    norm = text.strip().lower().rstrip(".")
    pending = get_pending_post()
    if pending:
        if norm in APPROVE_WORDS:
            clear_pending_post()
            if publish_post(pending):
                send_text(chat_id, "Опубліковано в канал ✅")
            else:
                send_text(chat_id, "Не вдалося опублікувати — перевір, чи бот є адміном каналу @RationalShelf.")
            return
        if norm in DENY_WORDS:
            clear_pending_post()
            send_text(chat_id, "Ок, не публікую.")
            return
        preview = pending if len(pending) <= 1100 else pending[:1100] + "…"
        send_text(chat_id, f"Є пост, що чекає на публікацію в {CHANNEL_NAME}. Щоб опублікувати — напиши «так, публікуй», щоб скасувати — «ні».\n\n{preview}")
        return

    if text.startswith("/start"):
        send_text(chat_id, "Привіт! Я Семія. Пиши запитання по книгах — я пошукаю в базі та підготую відповідь або пост.")
        return
    if sender not in ALLOWED:
        send_text(chat_id, "Це може зробити лише власник. Зверніться до @thediredixproject.")
        return
    reply = run_opencode(text, chat_id)
    chat_text, post = parse_reply(reply)
    wants_post = any(t in norm for t in POST_TRIGGERS)
    if post and not wants_post:
        chat_text = ((chat_text + "\n\n") if chat_text else "") + post
        post = None
    if post and any(bad in post.lower() for bad in LEAK_WORDS):
        chat_text = ((chat_text + "\n\n") if chat_text else "") + "[пост відхилено: містить службову інформацію]"
        post = None
    if chat_text and len(chat_text) > 2500:
        cut = chat_text.rfind("\n", 0, 2500)
        if cut < 1200:
            cut = 2500
        chat_text = chat_text[:cut] + "\n[відповідь скорочено мостом]"
    if chat_text:
        send_text(chat_id, chat_text)
    if post:
        set_pending_post(post)
        preview = post if len(post) <= 1100 else post[:1100] + "…"
        send_text(chat_id, f"Пост готовий. Опублікувати в {CHANNEL_NAME}? Напиши «так, публікуй» або «ні».\n\n{preview}")


def poll_forever():
    state = load_state()
    offset = state.get("offset", 0)
    print(f"[semia-bridge] start, allowed={sorted(ALLOWED)}, offset={offset}, model={MODEL}", flush=True)
    while True:
        try:
            r = call("getUpdates", {"offset": offset, "timeout": 30}, timeout=45)
            if not r.get("ok"):
                print(f"[getUpdates] {r.get('description')}", file=sys.stderr)
                time.sleep(3)
                continue
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                st = load_state()
                st["offset"] = offset
                save_state(st)
                if "message" in u:
                    handle_message(u["message"])
        except KeyboardInterrupt:
            print("[semia-bridge] stopped", flush=True)
            break
        except Exception as e:
            print(f"[poll error] {e}", file=sys.stderr)
            time.sleep(3)


if __name__ == "__main__":
    if not TOKEN:
        print("no TELEGRAM_BOT_TOKEN", file=sys.stderr)
        sys.exit(1)
    poll_forever()