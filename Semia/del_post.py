import json, urllib.request, sys

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
TOKEN = env["TELEGRAM_BOT_TOKEN"]
BASE = f"https://api.telegram.org/bot{TOKEN}"

def call(method, params):
    req = urllib.request.Request(BASE + "/" + method, data=json.dumps(params).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"ok": False, "description": str(e)}

for mid in [int(a) for a in sys.argv[1:]]:
    x = call("deleteMessage", {"chat_id": "-1003654061308", "message_id": mid})
    print(mid, x.get("ok"), x.get("description", ""))