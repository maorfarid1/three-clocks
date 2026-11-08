import re, sys
LLM_WORDS = r"\b(delve\w*|tapestry|intricate\w*|pivotal|paramount|underscor\w*|showcas\w*|realm|multifaceted|holistic|seamless\w*|harness\w*|unlock\w*|foster\w*|bolster\w*|testament|groundbreaking|cutting-edge|game-changer|ever-evolving|plethora|myriad|meticulous\w*|commendabl\w*|noteworthy|endeavou?r\w*|embark\w*|unveil\w*|in today's|it is worth noting|it is important to note|sheds? light|paves? the way|plays? a (crucial|pivotal|key) role)\b"
DRAMA = [r"\bthere is a (catch|twist|flip side|problem)\b", r"\bhere is (the|a) (catch|twist|thing)\b",
    r"\bwe take a step back\b", r"\bleaves? (two|three|several) things\b", r"\bon the table\b",
    r"\bin name only\b", r"\bmore than a toy\b", r"\bthe answer is (no|yes|simple)\b",
    r"\b(does|do) two things\b", r"\bnot merely\b", r"\bnot just\b", r"\bnot only\b",
    r"\bthe short answer\b", r"\bit turns out\b", r"\bin other words\b",
    r"\bremains? elusive\b", r"\bfar richer\b", r"\bworkhorse\b", r"\bas is often the case\b",
    r"\bthat is, ", r"\bbe it\b", r"\bthere is therefore an opportunity\b", r"\bthe review below\b",
    r"\btheir (purpose|aim) is\b", r"\bis (its|their) main advantage\b", r"\bwhich accounts for\b", r"\bthis leaves room\b",
    r"\brather than\b", r"\bnot a\b.{0,40}\bbut\b"]
def lint(t):
    out = []
    for m in re.finditer(r"[a-z\]\)][.!?,;:][A-Z]", t): out.append(("missing space", t[m.start()-15:m.end()+10]))
    for m in re.finditer(r"  +(?=\S)", t): out.append(("double space", t[max(0,m.start()-15):m.end()+10]))
    for m in re.finditer(r"\b(\w+)\s+\1\b", t, re.I):
        if m.group(1).lower() not in ("that", "had"): out.append(("repeated word", m.group(0)))
    for m in re.finditer(r"\s[,.;:]", t): out.append(("space before punctuation", t[max(0,m.start()-15):m.end()+5]))
    for m in re.finditer(r"[—–]|--", t): out.append(("dash", t[max(0,m.start()-20):m.end()+20]))
    for m in re.finditer(LLM_WORDS, t, re.I): out.append(("LLM word", m.group(0)))
    for p in DRAMA:
        for m in re.finditer(p, t, re.I): out.append(("drama/pivot", t[max(0,m.start()-20):m.end()+20]))
    w = len(re.findall(r"\w+", t)) or 1; cs = t.count(";") + t.count(":")
    if 1000*cs/w > 9: out.append(("too many colons/semicolons", f"{cs} in {w} words"))
    s = [x for x in re.split(r"(?<=[.!?])\s+", t) if len(x.split()) > 1]
    n = sum(1 for a, b in zip(s, s[1:]) if len(a.split()) < 9 and len(b.split()) < 9)
    if n: out.append(("back-to-back short sentences", f"{n} pair(s)"))
    for m in re.finditer(r"(?:^|(?<=[.!?]\s))([A-Z][^.!?]{0,60}\.)(?=\s*(\n|$))", t, re.M):
        if len(m.group(1).split()) < 10: out.append(("short zinger closing a paragraph", m.group(1)))
    try:
        from spellchecker import SpellChecker
        u = SpellChecker().unknown(re.findall(r"\b[a-z]{3,}\b", t))
        if u: out.append(("check spelling", ", ".join(sorted(u))))
    except ImportError: out.append(("note", "pip install pyspellchecker for spelling check"))
    return out
for f in sys.argv[1:]:
    r = lint(open(f).read()); print(f"== {f}: {len(r)} issue(s)")
    for k, v in r: print(f"  [{k}] {v}")
