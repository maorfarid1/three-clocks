"""Build refs.bib from the arXiv API records (data/arxiv_records.json) plus a
small set of DOI-verified non-arXiv items. Keys: firstauthor + year + first
content word of the title, e.g. chen2018neural."""
import json, re, unicodedata, os

HERE = os.path.dirname(os.path.abspath(__file__))
rec = json.load(open(os.path.join(HERE, "..", "data", "arxiv_records.json")))
# Published venues from the Semantic Scholar batch API (fetched 2026-11-07); the year stays the arXiv year.
VEN = json.load(open(os.path.join(HERE, "..", "data", "published_venues_s2.json")))

STOP = {"a", "an", "the", "on", "of", "in", "is", "are", "do", "how", "what", "when", "to", "and", "for", "via", "beyond", "towards", "toward", "understanding", "not", "all", "you", "need", "can", "at", "as", "with"}


def ascii(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def last(name):
    parts = name.replace(".", ". ").split()
    return parts[-1]


def key_for(authors, year, title):
    a = re.sub(r"[^a-z]", "", ascii(last(authors[0])).lower())
    words = [re.sub(r"[^a-z]", "", w.lower()) for w in ascii(title).split()]
    w = next((x for x in words if x and x not in STOP), "paper")
    return f"{a}{year}{w}"


def tex_escape(s):
    s = s.replace(" --- ", ": ").replace("\u2014", ", ").replace("\u2013", "-")
    s = s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    return s


def fmt_authors(authors):
    return " and ".join(tex_escape(a) for a in authors)


entries, keys = [], {}
for aid, r in sorted(rec.items()):
    k = key_for(r["authors"], r["year"], r["title"])
    base, n = k, 1
    while k in keys.values():
        n += 1
        k = f"{base}{chr(96 + n)}"
    keys[aid] = k
    title = tex_escape(r["title"])
    if aid in VEN:
        note = f",\n  note = {{Published in {tex_escape(VEN[aid])}}}"
    else:
        jr = r.get("journal") or ""
        note = f",\n  note = {{{tex_escape(jr)}}}" if jr else ""
    entries.append(
        f"@article{{{k},\n  title = {{{{{title}}}}},\n  author = {{{fmt_authors(r['authors'])}}},\n"
        f"  journal = {{arXiv preprint arXiv:{aid}}},\n  year = {{{r['year']}}}{note}\n}}\n")

MANUAL = r"""
@article{kramers1940brownian,
  title = {{Brownian motion in a field of force and the diffusion model of chemical reactions}},
  author = {Kramers, H. A.},
  journal = {Physica},
  volume = {7},
  number = {4},
  year = {1940},
  doi = {10.1016/S0031-8914(40)90098-2}
}
@article{hanggi1990reaction,
  title = {{Reaction-rate theory: fifty years after Kramers}},
  author = {H{\"a}nggi, Peter and Talkner, Peter and Borkovec, Michal},
  journal = {Reviews of Modern Physics},
  volume = {62},
  number = {2},
  year = {1990},
  doi = {10.1103/RevModPhys.62.251}
}
@book{strogatz2018nonlinear,
  title = {{Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering}},
  author = {Strogatz, Steven H.},
  edition = {2},
  publisher = {CRC Press},
  year = {2018},
  doi = {10.1201/9780429492563}
}
@article{sompolinsky1988chaos,
  title = {{Chaos in Random Neural Networks}},
  author = {Sompolinsky, H. and Crisanti, A. and Sommers, H. J.},
  journal = {Physical Review Letters},
  volume = {61},
  number = {3},
  year = {1988},
  doi = {10.1103/PhysRevLett.61.259}
}
@article{sussillo2013opening,
  title = {{Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks}},
  author = {Sussillo, David and Barak, Omri},
  journal = {Neural Computation},
  volume = {25},
  number = {3},
  year = {2013},
  doi = {10.1162/NECO_a_00409}
}
@article{e2017proposal,
  title = {{A Proposal on Machine Learning via Dynamical Systems}},
  author = {E, Weinan},
  journal = {Communications in Mathematics and Statistics},
  volume = {5},
  number = {1},
  year = {2017},
  doi = {10.1007/s40304-017-0103-z}
}
@article{pathak2018model,
  title = {{Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach}},
  author = {Pathak, Jaideep and Hunt, Brian and Girvan, Michelle and Lu, Zhixin and Ott, Edward},
  journal = {Physical Review Letters},
  volume = {120},
  number = {2},
  pages = {024102},
  year = {2018},
  doi = {10.1103/PhysRevLett.120.024102}
}
@book{vakakis2009nonlinear,
  title = {{Nonlinear Targeted Energy Transfer in Mechanical and Structural Systems}},
  author = {Vakakis, Alexander F. and Gendelman, Oleg V. and Bergman, Lawrence A. and McFarland, D. Michael and Kerschen, Gaetan and Lee, Young Sup},
  series = {Solid Mechanics and Its Applications},
  volume = {156},
  publisher = {Springer},
  year = {2009},
  doi = {10.1007/978-1-4020-9130-8}
}
@article{farid2022escape,
  title = {{Escape dynamics of a particle from a purely nonlinear truncated quartic potential well under harmonic excitation}},
  author = {Farid, Maor},
  journal = {Nonlinear Dynamics},
  year = {2022},
  doi = {10.1007/s11071-022-07976-5}
}
@article{farid2024unsupervised,
  title = {{Unsupervised data-driven response regime exploration and identification for dynamical systems}},
  author = {Farid, Maor},
  journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
  year = {2024},
  doi = {10.1063/5.0173938}
}
@article{farid2016tuned,
  title = {{Tuned pendulum as nonlinear energy sink for broad energy range}},
  author = {Farid, Maor and Gendelman, Oleg V.},
  journal = {Journal of Vibration and Control},
  year = {2016},
  doi = {10.1177/1077546315578561}
}
@article{farid2019escape,
  title = {{Escape of a forced-damped particle from weakly nonlinear truncated potential well}},
  author = {Farid, Maor and Gendelman, Oleg V.},
  journal = {arXiv preprint arXiv:1910.08545},
  year = {2019}
}
@misc{radford2019language,
  title = {{Language Models are Unsupervised Multitask Learners}},
  author = {Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  howpublished = {OpenAI technical report},
  year = {2019}
}
"""

with open(os.path.join(HERE, "refs.bib"), "w") as f:
    f.write("% Auto-generated by make_bib.py from arXiv API records and DOI-verified entries.\n\n")
    f.write("\n".join(entries))
    f.write(MANUAL)
json.dump(keys, open(os.path.join(HERE, "bibkeys.json"), "w"), indent=1)
print(len(entries), "arXiv entries")
