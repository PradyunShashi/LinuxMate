#!/usr/bin/env python3
"""
LinuxMate - LLM helper for Linux commands & OS lab work (clean JSON storage)
Team   : J. Kuladeep Adithya (24BCT0320), Pradyun Shashi (24BDS0052)
Course : Operating Systems - Case Study

Flask web app + Curated JSON Knowledge Base + Qwen2.5-1.5B-Instruct (LoRA fine-tuned)

  python linuxmate.py            start the web app  ->  http://localhost:5000
  python linuxmate.py --check    test Knowledge Base + model
"""
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import urllib.request

try:
    from flask import Flask, jsonify, request, send_file
    from werkzeug.exceptions import HTTPException
except ImportError as _e:
    sys.exit(f"Missing package ({_e}). Run: pip install flask")

BASE = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────── optional local Qwen + LoRA inference (no Ollama needed)
LORA_DIR      = os.path.join(BASE, "linuxmate-lora")
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MODEL_NAME = "linuxmate-qwen2.5-1.5b"

_local_model      = None   # PeftModel, loaded once at startup
_local_tokenizer  = None


def _try_load_local_model():
    """Load the fine-tuned Qwen + LoRA adapter if the adapter directory exists."""
    global _local_model, _local_tokenizer
    if not os.path.isdir(LORA_DIR):
        print(f"[Local] LoRA adapter not found at '{LORA_DIR}'. Falling back to Ollama / knowledge base.")
        return False
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        print(f"[Local] Loading {BASE_MODEL_ID} + LoRA adapter from '{LORA_DIR}' ...")
        dtype  = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        _local_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        if _local_tokenizer.pad_token is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        _local_model = PeftModel.from_pretrained(base_model, LORA_DIR)
        _local_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Local] Model ready on {device}. LinuxMate will use the fine-tuned Qwen adapter.")
        return True
    except Exception as exc:
        print(f"[Local] Could not load model: {exc}")
        return False


def _ask_local(messages):
    """Run inference using the locally loaded Qwen + LoRA model."""
    import torch
    text   = _local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _local_tokenizer(text, return_tensors="pt").to(_local_model.device)
    with torch.no_grad():
        output = _local_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True,
            pad_token_id=_local_tokenizer.eos_token_id,
        )
    gen_ids = output[0][inputs["input_ids"].shape[-1]:]
    return _local_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# =============================================================== settings
def _load_env():
    """Read .env next to this file. A real environment variable always wins."""
    path = os.path.join(BASE, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            os.environ.setdefault(k.strip(), v)


_load_env()
ENV = os.environ.get
OLLAMA = ENV("OLLAMA_URL", "auto").rstrip("/")   # "auto" = find Ollama (WSL or Windows)
MODEL = ENV("LINUXMATE_MODEL", "llama3.2:latest")
PORT = int(ENV("LINUXMATE_PORT", "5000"))

KNOWLEDGE_FILE = os.path.join(BASE, "knowledge.json")
HISTORY_FILE = os.path.join(BASE, "history.json")

RISKY = ["rm -rf", "rm -r", "mkfs", "dd if=", "chmod -r 777", "chmod 777", "shutdown", "reboot",
         "kill -9", "fdisk", "parted", "> /dev/sd"]


# =============================================================== storage (JSON)
def load_knowledge():
    """Load curated Q&A knowledge base from knowledge.json."""
    if not os.path.exists(KNOWLEDGE_FILE):
        return []
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history():
    """Load query history from history.json."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history_record(question, mode, answer, source):
    """Append a query to history.json and return its assigned ID."""
    hist = load_history()
    next_id = max([item.get("id", 0) for item in hist], default=0) + 1
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "id": next_id,
        "question": question[:1000],
        "mode": mode,
        "answer": answer[:3000],
        "source": source[:200],
        "created_at": now_str,
        "rating": 0
    }
    hist.append(record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)
    return next_id


def update_feedback(history_id, rating):
    """Update user feedback rating (+1 or -1) for a specific history ID."""
    hist = load_history()
    updated = False
    for item in hist:
        if item.get("id") == history_id:
            item["rating"] = rating
            updated = True
            break
    if updated:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2, ensure_ascii=False)
    return updated


def clear_history():
    """Clear all history items."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)


# ================================================================= Ollama Host
_host_cache = {}


def candidate_hosts():
    """localhost first, then the Windows host as seen from inside WSL."""
    cands = ["localhost", "127.0.0.1"]
    try:
        out = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=3).stdout
        m = re.search(r"default via (\S+)", out)
        if m:
            cands.append(m.group(1))
    except Exception:
        pass
    try:
        with open("/etc/resolv.conf") as f:
            cands += [ln.split()[1] for ln in f if ln.startswith("nameserver") and len(ln.split()) > 1]
    except Exception:
        pass
    return list(dict.fromkeys(cands))


def find_host(port):
    """First host that answers on this port (cached once found)."""
    if port in _host_cache:
        return _host_cache[port]
    for h in candidate_hosts():
        try:
            socket.create_connection((h, port), timeout=1.0).close()
            _host_cache[port] = h
            return h
        except OSError:
            continue
    return "127.0.0.1"


def ollama_url():
    """OLLAMA_URL=auto -> Ollama on localhost, or on the Windows host when running inside WSL."""
    if OLLAMA.lower() != "auto":
        return OLLAMA
    return f"http://{find_host(11434)}:11434"


# ================================================================= Ollama & Search
STOP = {"the", "a", "an", "of", "to", "in", "do", "i", "how", "what", "is", "my", "me", "for", "and", "can", "you"}


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOP}


def find_similar(question, mode, limit=3):
    """Rank knowledge rows: words shared with the saved question count triple."""
    q = words(question)
    knowledge = load_knowledge()
    scored = []
    for r in knowledge:
        if r.get("mode") != mode:
            continue
        q_words = words(r.get("question", ""))
        info_words = words((r.get("command", "") or "") + " " + (r.get("meaning", "") or ""))
        s = 3 * len(q & q_words) + len(q & info_words)
        if s > 0:
            scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def format_row(r):
    if r.get("mode") == "Concept":
        return r.get("meaning", "")
    text = f"Command: {r.get('command', '')}\nMeaning: {r.get('meaning', '')}"
    warn = r.get("warning")
    return text + (f"\nWarning: {warn}" if warn else "")


def build_messages(question, mode, examples):
    if mode == "Concept":
        system = ("You are LinuxMate, an Operating Systems lab helper. "
                  "Answer in 2 to 4 short sentences in plain language.")
    else:
        system = ("You are LinuxMate, a Linux command helper for students. "
                  "Reply in EXACTLY this format and nothing else:\n"
                  "Command: <one linux command>\nMeaning: <one short sentence>\n"
                  "Warning: <only add this line if the command can delete data or harm the system>")
    msgs = [{"role": "system", "content": system}]
    for r in examples:  # worked examples taken from Knowledge Base
        msgs += [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": format_row(r)}]
    return msgs + [{"role": "user", "content": question}]


def resolve_model():
    """Return the active model name: local Qwen adapter > Ollama > None."""
    # 1. Prefer the locally loaded fine-tuned adapter (no Ollama needed)
    if _local_model is not None:
        return LOCAL_MODEL_NAME
    # 2. Fall back to Ollama if the adapter is not available
    try:
        with urllib.request.urlopen(ollama_url() + "/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return None
    if not names:
        return None
    if MODEL in names:
        return MODEL
    return next((n for n in names if n.split(":")[0] == MODEL.split(":")[0]), names[0])


def ask_ollama(model, messages):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": 200}
    }).encode()
    req = urllib.request.Request(ollama_url() + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"].strip()


def add_warning(answer, mode):
    if mode == "Command" and any(w in answer.lower() for w in RISKY) and "warning:" not in answer.lower():
        answer += "\nWarning: This command can be destructive. Verify it before running."
    return answer


# ============================================================== web routes
app = Flask(__name__)


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):
        return e
    if request.path.startswith("/api/"):
        return jsonify(error="Server error: " + str(e)), 500
    raise e


@app.route("/")
def home():
    return send_file(os.path.join(BASE, "index.html"))


@app.route("/api/status")
def status():
    knowledge = load_knowledge()
    model = resolve_model()
    return jsonify({
        "oracle": True,       # backward compatibility with index.html indicator
        "db": True,
        "oracle_error": "",
        "ollama": model is not None,
        "model": model or MODEL,
        "local": _local_model is not None,
        "rows": len(knowledge)
    })


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()[:1000]
    mode = data.get("mode") if data.get("mode") in ("Command", "Concept") else "Command"
    if not question:
        return jsonify(error="Please type a question."), 400

    examples = find_similar(question, mode)
    res = {"mode": mode, "examples": [r["question"] for r in examples], "model": None}
    answer, model = None, resolve_model()

    # 1. Try locally loaded Qwen + LoRA adapter first (fastest, offline-capable)
    if _local_model is not None:
        try:
            answer = _ask_local(build_messages(question, mode, examples)) or None
            if answer:
                res["source"] = res["model"] = LOCAL_MODEL_NAME
                if data.get("compare"):
                    res["plain"] = add_warning(_ask_local(build_messages(question, mode, [])), mode)
        except Exception as e:
            answer, res["local_error"] = None, str(e)[:150]

    # 2. Fall back to Ollama if local model isn't loaded or inference failed
    elif model:
        try:
            answer = ask_ollama(model, build_messages(question, mode, examples)) or None
            if answer:
                res["source"] = res["model"] = model
                if data.get("compare"):
                    res["plain"] = add_warning(ask_ollama(model, build_messages(question, mode, [])), mode)
        except Exception as e:
            answer, res["ollama_error"] = None, str(e)[:150]

    # Fallback to curated knowledge base if Ollama is offline or call failed
    if answer is None:
        if examples:
            answer, res["source"] = format_row(examples[0]), "knowledge_base"
        else:
            answer, res["source"] = "Sorry, I could not find a match. Start Ollama to get generated answers.", "none"

    res["answer"] = add_warning(answer, mode)
    res["risky"] = "warning:" in res["answer"].lower()
    res["history_id"] = save_history_record(question, mode, res["answer"], res["source"])
    res["db_error"] = None
    return jsonify(res)


@app.route("/api/library")
def library():
    q = request.args.get("q", "").strip().lower()
    mode = request.args.get("mode", "")
    items = load_knowledge()

    results = []
    for r in items:
        if mode in ("Command", "Concept") and r.get("mode") != mode:
            continue
        if q:
            text = (r.get("question", "") + " " + r.get("command", "") + " " + r.get("meaning", "")).lower()
            if q not in text:
                continue
        results.append({
            "id": r.get("id"),
            "category": r.get("category"),
            "mode": r.get("mode"),
            "question": r.get("question"),
            "command": r.get("command"),
            "meaning": r.get("meaning"),
            "warning": r.get("warning")
        })
    return jsonify(results)


@app.route("/api/history", methods=["GET", "DELETE"])
def history():
    if request.method == "DELETE":
        clear_history()
        return jsonify(ok=True)
    hist = load_history()
    # return last 50, newest first
    recent = list(reversed(hist))[:50]
    return jsonify([{
        "id": r.get("id"),
        "question": r.get("question"),
        "mode": r.get("mode"),
        "answer": r.get("answer"),
        "source": r.get("source"),
        "created_at": r.get("created_at")
    } for r in recent])


@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True, silent=True) or {}
    try:
        hid = int(data.get("history_id"))
        rating = 1 if int(data.get("rating", 1)) > 0 else -1
        update_feedback(hid, rating)
    except Exception as e:
        return jsonify(error="Could not save feedback: " + str(e)), 400
    return jsonify(ok=True)


@app.route("/api/stats")
def stats():
    hist = load_history()
    likes = sum(1 for h in hist if h.get("rating") == 1)
    dislikes = sum(1 for h in hist if h.get("rating") == -1)
    return jsonify({
        "questions": len(hist),
        "likes": likes,
        "dislikes": dislikes
    })


# ============================================================ setup checker
def check():
    def line(ok, msg):
        print(("  [ OK ] " if ok else "  [FAIL] ") + msg)

    print("\n1) Knowledge Base")
    k = load_knowledge()
    if len(k) > 0:
        line(True, f"Loaded {len(k)} Q&A pairs from knowledge.json")
    else:
        line(False, "knowledge.json is empty or missing")

    print("\n2) Local Qwen + LoRA Model")
    if _local_model is not None:
        line(True, f"Fine-tuned adapter loaded from '{LORA_DIR}'")
        try:
            reply = _ask_local([{"role": "user", "content": "Say: ready"}])
            line(True, "Model replied: " + repr(reply[:50]))
        except Exception as e:
            line(False, "Model did not answer: " + str(e))
    else:
        line(False, f"LoRA adapter not found at '{LORA_DIR}'. Run: python train_lora.py")
        print("\n3) Ollama (fallback)")
        model = resolve_model()
        if model is None:
            line(False, f"No model reachable at {ollama_url()} -> Download Ollama from https://ollama.com and run: ollama run llama3.2")
            print("         (Note: LinuxMate can still run right now using knowledge.json fallback answers!)")
        else:
            line(True, f"Ollama reachable, LinuxMate will use: {model}")
            try:
                line(True, "Model replied: " + repr(ask_ollama(model, [{"role": "user", "content": "Say: ready"}])[:50]))
            except Exception as e:
                line(False, "Model did not answer: " + str(e))

    print("\n" + ("Everything is ready -> python linuxmate.py\n" if len(k) > 0 else "Fix the [FAIL] lines, then run --check again.\n"))


if __name__ == "__main__":
    if "--check" in sys.argv:
        _try_load_local_model()
        check()
        sys.exit(0)
    print("=" * 62)
    k_len = len(load_knowledge())
    print(f" LinuxMate | Knowledge Base: {k_len} items")
    print("=" * 62)
    _try_load_local_model()
    active_model = LOCAL_MODEL_NAME if _local_model is not None else MODEL
    print(f" Active model : {active_model}")
    print(f" Open http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
