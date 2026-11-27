#!/usr/bin/env python3
"""Print exp-035's tables (markdown) from the committed metrics.json. No computation beyond
formatting and counting: every number printed is read off the file.

    python experiments/exp-035/summarize.py experiments/exp-035/metrics.json
"""
from __future__ import annotations

import json
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
arms = list(m["inputs"])
strata = ["overall", "2", "3", "4"]


def f(x):
    return "—" if x is None else f"{x:.4f}"


print("## Overall, per variant (EM / SARI / GED; GED dropped)\n")
print("| arm | hg_fork EM | SARI | GED | dropped | paper-literal EM | SARI | GED | repo-port EM | SARI | GED |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for a in arms:
    h = m["rescored"]["hg_fork"][a]["overall"]
    l = m["rescored"]["hg_paper_literal"][a]["overall"]
    s = m["stored_repo_port"][a]["overall"]
    print(f"| {a} | {f(h['break_exact_match'])} | {f(h['sari'])} | {f(h['ged'])} | {h['ged_dropped']} | "
          f"{f(l['break_exact_match'])} | {f(l['sari'])} | {f(l['ged'])} | "
          f"{f(s['break_exact_match'])} | {f(s['sari'])} | {f(s['ged'])} |")

print("\n## hg_fork per gold hop (EM / SARI / GED)\n")
print("| arm | hop | n | EM | SARI | GED | GED dropped |")
print("|---|---|---|---|---|---|---|")
for a in arms:
    for st in strata[1:]:
        h = m["rescored"]["hg_fork"][a][st]
        print(f"| {a} | {st} | {h['n']} | {f(h['break_exact_match'])} | {f(h['sari'])} | {f(h['ged'])} | {h['ged_dropped']} |")

print("\n## Stored (exp-030) under / over-decomposition per gold hop (descriptive)\n")
print("| arm | under 2 | under 3 | under 4 | over 2 | over 3 | over 4 |")
print("|---|---|---|---|---|---|---|")
for a in arms:
    s = m["stored_repo_port"][a]
    print(f"| {a} | " + " | ".join(f(s[h]["under_decomposition"]) for h in strata[1:]) + " | "
          + " | ".join(f(s[h]["over_decomposition"]) for h in strata[1:]) + " |")

fam = m["paired_family"]
print(f"\n## Paired family (hg_fork; {fam['size']} tests, {fam['size_with_p']} with p; Holm)\n")
print("| A vs B | stratum | metric | A | B | diff | p | p_holm | sig (uncorr) | sig (Holm) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for t in fam["tests"]:
    a_val = t.get("system_a_rate", t.get("system_a"))
    b_val = t.get("system_b_rate", t.get("system_b"))
    p = t.get("p_value")
    print(f"| {t['system_a']} vs {t['system_b']} | {t['stratum']} | {t['metric']} | {f(a_val)} | {f(b_val)} | "
          f"{f(t.get('difference'))} | {'—' if p is None else f'{p:.3g}'} | "
          f"{'—' if t['p_holm'] is None else f'{t['p_holm']:.3g}'} | {t['significant']} | {t['significant_holm']} |")
sig_u = sum(1 for t in fam["tests"] if t["significant"])
sig_h = sum(1 for t in fam["tests"] if t["significant_holm"])
print(f"\nsignificant uncorrected: {sig_u}/{fam['size']}; Holm survivors: {sig_h}/{fam['size']}")
print(f"wall_seconds_total: {m['wall_seconds_total']:.1f}")
print("environment:", json.dumps(m["environment"]))
