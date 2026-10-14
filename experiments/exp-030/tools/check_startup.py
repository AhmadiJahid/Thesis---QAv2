"""Startup-line checker for one exp-030 arm. Exit 0 = all checks pass, 1 = mismatch, 2 = timeout/dead."""
import json, re, sys, time, os
log, quant, params, pid = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
SHA = "98e4f7201b0b11cabfd9d1df9ee154464ebc230b7296c27be98ae02cd1efa114"
def alive():
    try: os.kill(pid, 0); return True
    except OSError: return False
def read():
    try: return open(log, errors="replace").read()
    except FileNotFoundError: return ""
deadline = time.time() + 1800
res = {}
while time.time() < deadline:
    t = read()
    m = re.search(r"^Evaluation set: (\{.*\})\s*$", t, re.M)
    ms = re.search(r"^\[model_size\] .*parameters=([\d,]+)", t, re.M)
    if m and "eval" not in res:
        ev = json.loads(m.group(1))
        res["eval"] = (ev.get("named_eval_set") == "musique_dev_full" and ev.get("matches_named_set") is True
                       and ev.get("rows_loaded_total") == 2417
                       and ev.get("rows_loaded_per_hop") == {"2": 1252, "3": 760, "4": 405})
        res["sha"] = f'"input_sha256": "{SHA}"' in t
        res["quant"] = re.search(r'^  "quantization": "([^"]+)"', t, re.M).group(1) == quant
        res["ordering"] = re.search(r'^  "few_shot_ordering": "([^"]+)"', t, re.M).group(1) == "as_is"
        print("startup checks:", res, flush=True)
        if not all(res.values()): sys.exit(1)
    if ms:
        res["model_size"] = ms.group(1) == params
        print("model_size:", ms.group(0), "expected", params, "->", res["model_size"], flush=True)
        sys.exit(0 if all(res.values()) else 1)
    if not alive():
        print("process exited before checks completed", res, flush=True); sys.exit(2)
    time.sleep(10)
print("timeout", res, flush=True); sys.exit(2)
