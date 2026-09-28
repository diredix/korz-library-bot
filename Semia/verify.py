import json, urllib.request, urllib.error, sys

def load_env():
    env = {}
    with open(r"C:\Users\diredix\Downloads\books_qdrant\Semia\.env", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

env = load_env()
TOKEN = env.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    print("no token"); sys.exit(1)

BASE = f"https://api.telegram.org/bot{TOKEN}"

def call(method, params=None):
    url = BASE + "/" + method
    data = json.dumps(params or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())

me = call("getMe")
print("me:", me.get("result", {}).get("username"), "| bot:", me.get("result", {}).get("first_name"))

ups = call("getUpdates", {"timeout": 0})
print("updates ok:", ups.get("ok"), "| count:", len(ups.get("result", [])))
for u in ups.get("result", [])[-10:]:
    msg = u.get("message") or u.get("channel_post") or {}
    chat = msg.get("chat", {})
    text = (msg.get("text") or msg.get("caption") or "[none]")[:80]
    print(f"  chat: {chat.get('id')} ({chat.get('type')} @{chat.get('username', '')}) from={msg.get('from',{}).get('id','')} text={text!r}")

for target in ["-1003654061308", "@rationalshelf2", "@RationalShelf"]:
    try:
        r = call("getChat", {"chat_id": target})
        if r.get("ok"):
            res = r["result"]
            print(f"getChat {target}: OK id={res.get('id')} type={res.get('type')} title={res.get('title')!r} @{res.get('username','')}")
        else:
            print(f"getChat {target}: FAIL {r.get('description')}")
    except Exception as e:
        print(f"getChat {target}: EXC {e}")