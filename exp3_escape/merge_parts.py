"""Merge escape.py outputs run on disjoint prompt ranges into one file."""
import json, sys
out, parts = sys.argv[1], sys.argv[2:]
D = [json.load(open(p)) for p in parts]
m = D[0]
for d in D[1:]:
    m["loops"] += d["loops"]; m["mc"] += d["mc"]
m["loops"].sort(key=lambda l: l["prompt"])
m["args"]["first_prompt"] = 0; m["args"]["n_prompts"] = max(d["args"]["n_prompts"] for d in D); m["args"]["parts"] = parts
json.dump(m, open(out, "w"))
print(out, len(m["loops"]), "loops")
