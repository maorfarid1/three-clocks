import sys, os, re
for f in sys.argv[1:]:
    paras = [p for p in re.split(r"\n\s*\n", open(f).read()) if p.strip()]
    chunks, cur = [], []
    for p in paras:
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= 400:
            chunks.append(cur); cur = []
    if cur:
        if chunks and sum(len(x.split()) for x in cur) < 150: chunks[-1] += cur
        else: chunks.append(cur)
    base = os.path.splitext(f)[0]
    for i, c in enumerate(chunks):
        open(f"{base}.part{i+1}.txt", "w").write("\n\n".join(c))
        print(f"{base}.part{i+1}.txt", sum(len(x.split()) for x in c))
