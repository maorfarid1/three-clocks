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
@article{bennett2002huygens,
  title = {{Huygens's clocks}},
  author = {Bennett, Matthew and Schatz, Michael F. and Rockwood, Heidi and Wiesenfeld, Kurt},
  journal = {Proceedings of the Royal Society of London. Series A},
  volume = {458},
  number = {2019},
  year = {2002},
  doi = {10.1098/rspa.2001.0888}
}
@article{pantaleone2002synchronization,
  title = {{Synchronization of metronomes}},
  author = {Pantaleone, James},
  journal = {American Journal of Physics},
  volume = {70},
  number = {10},
  year = {2002},
  doi = {10.1119/1.1501118}
}
@incollection{kuramoto1975self,
  title = {{Self-entrainment of a population of coupled non-linear oscillators}},
  author = {Kuramoto, Yoshiki},
  booktitle = {International Symposium on Mathematical Problems in Theoretical Physics},
  series = {Lecture Notes in Physics},
  volume = {39},
  publisher = {Springer},
  year = {1975},
  doi = {10.1007/BFb0013365}
}
@article{acebron2005kuramoto,
  title = {{The Kuramoto model: A simple paradigm for synchronization phenomena}},
  author = {Acebr{\'o}n, Juan A. and Bonilla, L. L. and P{\'e}rez Vicente, Conrad J. and Ritort, F{\'e}lix and Spigler, Renato},
  journal = {Reviews of Modern Physics},
  volume = {77},
  number = {1},
  year = {2005},
  doi = {10.1103/RevModPhys.77.137}
}
@article{goldreich1966spin,
  title = {{Spin-orbit coupling in the solar system}},
  author = {Goldreich, Peter and Peale, Stanton},
  journal = {The Astronomical Journal},
  volume = {71},
  pages = {425},
  year = {1966},
  doi = {10.1086/109947}
}
@article{may1976simple,
  title = {{Simple mathematical models with very complicated dynamics}},
  author = {May, Robert M.},
  journal = {Nature},
  volume = {261},
  year = {1976},
  doi = {10.1038/261459a0}
}
@article{feigenbaum1978quantitative,
  title = {{Quantitative universality for a class of nonlinear transformations}},
  author = {Feigenbaum, Mitchell J.},
  journal = {Journal of Statistical Physics},
  volume = {19},
  year = {1978},
  doi = {10.1007/BF01020332}
}
@article{lorenz1963deterministic,
  title = {{Deterministic nonperiodic flow}},
  author = {Lorenz, Edward N.},
  journal = {Journal of the Atmospheric Sciences},
  volume = {20},
  year = {1963},
  doi = {10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2}
}
@article{mcdonald1985fractal,
  title = {{Fractal basin boundaries}},
  author = {McDonald, Steven W. and Grebogi, Celso and Ott, Edward and Yorke, James A.},
  journal = {Physica D: Nonlinear Phenomena},
  volume = {17},
  number = {2},
  year = {1985},
  doi = {10.1016/0167-2789(85)90001-6}
}
@article{thompson1990fractal,
  title = {{Fractal control boundaries of driven oscillators and their relevance to safe engineering design}},
  author = {Thompson, J. M. T. and Soliman, M. S.},
  journal = {Proceedings of the Royal Society of London. Series A},
  volume = {428},
  number = {1874},
  year = {1990},
  doi = {10.1098/rspa.1990.0022}
}
@article{thompson1997designing,
  title = {{Designing against capsize in beam seas: Recent advances and new insights}},
  author = {Thompson, J. M. T.},
  journal = {Applied Mechanics Reviews},
  volume = {50},
  number = {5},
  year = {1997},
  doi = {10.1115/1.3101710}
}
@article{arrhenius1889reaktionsgeschwindigkeit,
  title = {{{\"U}ber die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch S{\"a}uren}},
  author = {Arrhenius, Svante},
  journal = {Zeitschrift f{\"u}r Physikalische Chemie},
  volume = {4U},
  number = {1},
  year = {1889},
  doi = {10.1515/zpch-1889-0416}
}
@article{koopman1931hamiltonian,
  title = {{Hamiltonian systems and transformation in Hilbert space}},
  author = {Koopman, B. O.},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {17},
  number = {5},
  year = {1931},
  doi = {10.1073/pnas.17.5.315}
}
@article{mezic2005spectral,
  title = {{Spectral properties of dynamical systems, model reduction and decompositions}},
  author = {Mezi{\'c}, Igor},
  journal = {Nonlinear Dynamics},
  volume = {41},
  year = {2005},
  doi = {10.1007/s11071-005-2824-x}
}
@article{rowley2009spectral,
  title = {{Spectral analysis of nonlinear flows}},
  author = {Rowley, Clarence W. and Mezi{\'c}, Igor and Bagheri, Shervin and Schlatter, Philipp and Henningson, Dan S.},
  journal = {Journal of Fluid Mechanics},
  volume = {641},
  year = {2009},
  doi = {10.1017/S0022112009992059}
}
@misc{qwen2025qwen25,
  title = {{Qwen2.5 Technical Report}},
  author = {{Qwen Team}},
  howpublished = {arXiv preprint arXiv:2412.15115},
  year = {2024}
}
@misc{keskar2019ctrl,
  title = {{CTRL: A Conditional Transformer Language Model for Controllable Generation}},
  author = {Keskar, Nitish Shirish and McCann, Bryan and Varshney, Lav R. and Xiong, Caiming and Socher, Richard},
  howpublished = {arXiv preprint arXiv:1909.05858},
  year = {2019}
}
@misc{yang2025qwen3,
  title = {{Qwen3 Technical Report}},
  author = {Yang, An and Li, Anfeng and Yang, Baosong and others},
  howpublished = {arXiv preprint arXiv:2505.09388},
  year = {2025}
}
@article{guo2025deepseekr1,
  title = {{DeepSeek-R1} Incentivizes Reasoning in {LLMs} through Reinforcement Learning},
  author = {{DeepSeek-AI} and Guo, Daya and Yang, Dejian and Zhang, Haowei and others},
  journal = {Nature},
  volume = {645},
  year = {2025},
  doi = {10.1038/s41586-025-09422-z},
  note = {arXiv:2501.12948}
}
@misc{cobbe2021training,
  title = {{Training Verifiers to Solve Math Word Problems}},
  author = {Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and others},
  howpublished = {arXiv preprint arXiv:2110.14168},
  year = {2021}
}
@misc{deepseekr1card,
  title = {{DeepSeek-R1} model card, Usage Recommendations},
  author = {{DeepSeek-AI}},
  howpublished = {\url{https://huggingface.co/deepseek-ai/DeepSeek-R1}, accessed November 2026},
  year = {2025}
}
@misc{qwen3card,
  title = {{Qwen3-0.6B} model card, Best Practices},
  author = {{Qwen Team}},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen3-0.6B}, accessed November 2026},
  year = {2025}
}
@article{eckmann1987recurrence,
  title = {Recurrence Plots of Dynamical Systems},
  author = {Eckmann, Jean-Pierre and Oliffson Kamphorst, Sylvie and Ruelle, David},
  journal = {Europhysics Letters},
  volume = {4},
  number = {9},
  year = {1987},
  doi = {10.1209/0295-5075/4/9/006}
}
@article{marwan2007recurrence,
  title = {Recurrence Plots for the Analysis of Complex Systems},
  author = {Marwan, Norbert and Romano, M. Carmen and Thiel, Marco and Kurths, J{\"u}rgen},
  journal = {Physics Reports},
  volume = {438},
  year = {2007},
  doi = {10.1016/j.physrep.2006.11.001}
}
@misc{pham2026mind,
  title = {Is my model ``mind blurting''? {I}nterpreting the dynamics of reasoning tokens with Recurrence Quantification Analysis ({RQA})},
  author = {Pham, Quoc Tuan and Jafari, Mehdi and Salim, Flora},
  howpublished = {arXiv preprint arXiv:2602.06266},
  year = {2026}
}
}
@misc{turner2023steering,
  title = {Steering Language Models With Activation Engineering},
  author = {Turner, Alexander Matt and Thiergart, Lisa and Leech, Gavin and Udell, David and others},
  howpublished = {arXiv preprint arXiv:2308.10248},
  year = {2023}
}
@misc{zou2023representation,
  title = {Representation Engineering: A Top-Down Approach to {AI} Transparency},
  author = {Zou, Andy and Phan, Long and Chen, Sarah and Campbell, James and others},
  howpublished = {arXiv preprint arXiv:2310.01405},
  year = {2023}
}
@misc{pipis2025wait,
  title = {{Wait, Wait, Wait... Why Do Reasoning Models Loop?}},
  author = {Pipis, Charilaos and Garg, Shivam and Kontonis, Vasilis and Shrivastava, Vaishnavi and Krishnamurthy, Akshay and Papailiopoulos, Dimitris},
  howpublished = {arXiv preprint arXiv:2512.12895},
  year = {2025}
}
@misc{litman2026origin,
  title = {{The Origin of Edge of Stability}},
  author = {Litman, Elon},
  howpublished = {arXiv preprint arXiv:2604.20446},
  year = {2026}
}
@misc{cohen2024central,
  title = {{Understanding Optimization in Deep Learning with Central Flows}},
  author = {Cohen, Jeremy M. and Damian, Alex and Talwalkar, Ameet and Kolter, J. Zico and Lee, Jason D.},
  howpublished = {arXiv preprint arXiv:2410.24206},
  year = {2024}
}
@misc{agarwala2022secondorder,
  title = {{Second-order regression models exhibit progressive sharpening to the edge of stability}},
  author = {Agarwala, Atish and Pedregosa, Fabian and Pennington, Jeffrey},
  howpublished = {arXiv preprint arXiv:2210.04860},
  year = {2022}
}
@misc{kalra2023universal,
  title = {{Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos}},
  author = {Kalra, Dayal Singh and He, Tianyu and Barkeshli, Maissam},
  howpublished = {arXiv preprint arXiv:2311.02076},
  year = {2023}
}
@misc{song2023trajectory,
  title = {{Trajectory Alignment: Understanding the Edge of Stability Phenomenon via Bifurcation Theory}},
  author = {Song, Minhak and Yun, Chulhee},
  howpublished = {arXiv preprint arXiv:2307.04204},
  year = {2023}
}
@misc{mulayoff2026stability,
  title = {{On the Stability of Nonlinear Dynamics in GD and SGD: Beyond Quadratic Potentials}},
  author = {Mulayoff, Rotem and Stich, Sebastian U.},
  howpublished = {arXiv preprint arXiv:2602.14789},
  year = {2026}
}
@misc{gan2026bifurcation,
  title = {{A Bifurcation Theory Framework for Gradient Descent on the Edge of Stability}},
  author = {Gan, Eric},
  howpublished = {arXiv preprint arXiv:2606.15551},
  year = {2026}
}
@misc{berthier2026multiple,
  title = {{The Multiple Timescales of Gradient Descent on the Edge of Stability: A Perturbative Derivation of the Central Flow}},
  author = {Berthier, Rapha{\"e}l},
  howpublished = {arXiv preprint arXiv:2609.01034},
  year = {2026}
}
@misc{marion2026edge,
  title = {{Edge Flow: A Tractable and Predictive Continuous-Time Model for Gradient Descent at the Edge of Stability}},
  author = {Marion, Pierre},
  howpublished = {arXiv preprint arXiv:2606.18080},
  year = {2026}
}
"""

with open(os.path.join(HERE, "refs.bib"), "w") as f:
    f.write("% Auto-generated by make_bib.py from arXiv API records and DOI-verified entries.\n\n")
    f.write("\n".join(entries))
    f.write(MANUAL)
json.dump(keys, open(os.path.join(HERE, "bibkeys.json"), "w"), indent=1)
print(len(entries), "arXiv entries")
