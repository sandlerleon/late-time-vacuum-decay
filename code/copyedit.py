# -*- coding: utf-8 -*-
"""Copy-editing conventions applied to every prose block of the manuscript and supplement at build time (from the Rubriq language review of
30 October 2026, accepted selectively): American spelling, "rethermalize", "approximately" before a number, a comma after "i.e." and "e.g.",
closed "non-" compounds. Mathematics ($...$), citations (\\cite{...}), URLs and file names are left untouched. Rubriq edits that changed the
meaning or broke terminology (for example "vacua" -> "vacuoles", "big bang" -> "large bang", "sub-Planckian" -> "subPlanckian", "stress-energy" ->
"stress energy") were not accepted.
"""
import re

WORDS = [
    (r"\bre-thermaliz", "rethermaliz"), (r"\bRe-thermaliz", "Rethermaliz"),
    (r"\bcentred\b", "centered"), (r"\bcentre\b", "center"), (r"\bcentres\b", "centers"),
    (r"\bbehaviour", "behavior"), (r"\bmillimetre", "millimeter"), (r"\blicence\b", "license"), (r"\banalogue\b", "analog"), (r"\bgrey\b", "gray"),
    (r"\b([Dd]e)?[Ss]tabilis(e|ed|es|ing|ation)\b", lambda m: m.group(0).replace("abilis", "abiliz")),
    (r"\bthermalis(e|ed|es|ing|ation)\b", lambda m: m.group(0).replace("alis", "aliz")),
    (r"\bsummaris(e|ed|es|ing)\b", lambda m: m.group(0).replace("aris", "ariz")),
    (r"\bnon-relativistic\b", "nonrelativistic"), (r"\bnon-thermal\b", "nonthermal"), (r"\bnon-standard\b", "nonstandard"),
    (r"\bnon-restoration\b", "nonrestoration"), (r"\bultra-relativistic\b", "ultrarelativistic"),
    (r"\bi\.e\.(?!,)", "i.e.,"), (r"\be\.g\.(?!,)", "e.g.,"),
]
ABOUT = (re.compile(r"\babout(?= (?:[0-9]|\$|~|−|one |two |three |four |five |six |ten |a third|half))"), "approximately")
_PROTECT = re.compile(r"(\$[^$]*\$|\\cite\{[^}]*\}|https?://\S+|\b[\w\-]+\.(?:py|json|csv|docx|tex|pdf)\b)")


def ce(text):
    if not isinstance(text, str) or not text:
        return text
    text = ABOUT[0].sub(ABOUT[1], text)                      # before splitting: "about" is often followed by inline mathematics
    parts = _PROTECT.split(text)
    for k in range(0, len(parts), 2):                       # even indices are prose, odd are protected
        s = parts[k]
        for pat, rep in WORDS:
            s = re.sub(pat, rep, s)
        parts[k] = s
    return "".join(parts)


def block(b):
    k = b[0]
    if k in ("p", "abstract", "note", "h1", "h2", "keywords"):
        return (k, ce(b[1])) + tuple(b[2:])
    if k == "bullets":
        return (k, [ce(x) for x in b[1]]) + tuple(b[2:])
    if k == "pb":
        return (k, ce(b[1]), ce(b[2])) + tuple(b[3:])
    if k == "table":
        return (k, ce(b[1]), [ce(str(c)) for c in b[2]], [[ce(str(c)) for c in row] for row in b[3]]) + tuple(b[4:])
    if k == "fig":
        return (k, b[1], ce(b[2])) + tuple(b[3:])
    return b


if __name__ == "__main__":
    for t in ("it is re-thermalized at about 5 MeV, i.e. the centre", "a statement about the geometry", "$\\rm about$ and the bubble_handoff.py file"):
        print(t, "->", ce(t))
