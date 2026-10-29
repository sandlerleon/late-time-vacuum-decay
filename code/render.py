# -*- coding: utf-8 -*-
"""Single-source renderer: the paper is written once as a list of blocks with LaTeX-style inline math, and rendered both to a Word
document (inline math converted to Unicode) and to a LaTeX file.

Block types
    ("title", text) ("author", text) ("h1", text) ("h2", text) ("p", text) ("eq", latex) ("fig", file, caption) ("table", caption, header, rows, widths)
    ("abstract", text) ("keywords", text) ("bullets", [text, ...]) ("ref", text) ("note", text)
Inline math is written between dollar signs in LaTeX syntax; display equations are LaTeX and are numbered automatically.
"""
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_TAB_ALIGNMENT
from docx.shared import Inches, Pt
import omml as OM

# ----------------------------------------------------------------------------------------------- LaTeX -> Unicode
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ",
         "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ",
         "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Lambda": "Λ", "Sigma": "Σ", "Omega": "Ω", "Phi": "Φ", "ell": "ℓ", "hbar": "ℏ", "partial": "∂",
         "sim": "~", "simeq": "≃", "approx": "≈", "lesssim": "≲", "gtrsim": "≳", "ll": "≪", "gg": "≫", "times": "×", "cdot": "·", "to": "→", "pm": "±",
         "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "infty": "∞", "int": "∫", "sum": "∑", "propto": "∝", "equiv": "≡", "dots": "…", "ldots": "…",
         "prime": "′", "quad": "  ", "qquad": "    ", ",": " ", ";": " ", "!": "", "mathrm": "", "rm": "", "text": "", "mathcal": "", "left": "", "right": "",
         "Big": "", "big": "", "nabla": "∇", "ln": "ln", "log": "log", "exp": "exp", "cos": "cos", "sin": "sin", "arcsin": "arcsin", "max": "max", "min": "min",
         "langle": "⟨", "rangle": "⟩", "mid": "|", "perp": "⊥", "star": "*", "bar": "̄", "tilde": "̃", "hat": "̂", "ast": "*", "Longrightarrow": "⇒"}
SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "−": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
       "n": "ⁿ", "i": "ⁱ", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "k": "ᵏ", "m": "ᵐ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ",
       "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "x": "ˣ", "y": "ʸ", "z": "ᶻ"}
SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "−": "₋", "=": "₌", "(": "₍", ")": "₎",
       "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ"}


def _group(s, i):
    """Return (content, next_index) for a {...} group or a single token starting at s[i]."""
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 1, i + 1
        while j < len(s) and depth:
            depth += (s[j] == "{") - (s[j] == "}")
            j += 1
        return s[i + 1:j - 1], j
    if s[i] == "\\":
        m = re.match(r"\\[A-Za-z]+", s[i:])
        if m:
            return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def tex2uni(s):
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            name = m.group(1)
            i += len(m.group(0))
            if name == "frac":
                a, i = _group(s, i)
                b, i = _group(s, i)
                ta, tb = tex2uni(a), tex2uni(b)
                wrap = lambda t: t if re.fullmatch(r"[\w.′]+", t) else "(" + t + ")"
                out.append(wrap(ta) + "/" + wrap(tb))
            elif name == "sqrt":
                a, i = _group(s, i)
                out.append("√(" + tex2uni(a) + ")" if len(a) > 1 else "√" + tex2uni(a))
            elif name in ("mathrm", "text", "rm", "mathcal", "mathbf", "operatorname"):
                a, i = _group(s, i)
                out.append(tex2uni(a))
            elif name in GREEK:
                out.append(GREEK[name])
            elif name in ("\\",):
                out.append(" ")
            elif name in ("{", "}", "%", "&", "_", "$", "#"):
                out.append(name)
            else:
                out.append(name)
        elif ch in "^_":
            a, i = _group(s, i + 1)
            t = tex2uni(a)
            table = SUP if ch == "^" else SUB
            if t and all(c in table for c in t):
                out.append("".join(table[c] for c in t))
            else:
                out.append(("^" if ch == "^" else "_") + ("(" + t + ")" if len(t) > 1 else t))
        elif ch in "{}":
            i += 1
        elif ch == "~":
            out.append(" ")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def convert_inline(text):
    """Replace $...$ in text by Unicode; returns a list of (string, is_math)."""
    parts = re.split(r"(\$[^$]+\$)", text)
    res = []
    for p in parts:
        if p.startswith("$") and p.endswith("$") and len(p) > 1:
            res.append((tex2uni(p[1:-1]), True))
        elif p:
            res.append((p, False))
    return res


# ------------------------------------------------------------------------------------------------------------ Word
def _shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def _borders(table, color="888888"):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge)
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


def render_docx(blocks, path, fig_dir, font="Times New Roman", base=11):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = font
    st.element.rPr.rFonts.set(qn("w:eastAsia"), font)
    st.font.size = Pt(base)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.15
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin = s.bottom_margin = Inches(1.0)
    eq_n = 0

    def add_text(p, text, italic=False, bold=False, size=None):
        for seg in re.split(r"(\$[^$]+\$)", text):
            if not seg:
                continue
            if seg.startswith("$") and seg.endswith("$") and len(seg) > 1:
                om = OM.tex_to_omml(seg[1:-1])
                if size:
                    OM.set_size(om, int(round(size * 2)))
                p._p.append(om)
            else:
                r = p.add_run(seg.replace("---", "—").replace("--", "–"))
                r.italic = italic
                r.bold = bold
                if size:
                    r.font.size = Pt(size)

    for b in blocks:
        kind = b[0]
        if kind == "title":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_text(p, b[1], bold=True, size=16); p.paragraph_format.space_after = Pt(10)
        elif kind == "author":
            for line in b[1].split("\n"):
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_text(p, line, size=10.5); p.paragraph_format.space_after = Pt(1)
        elif kind == "abstract":
            p = doc.add_paragraph(); add_text(p, "Abstract", bold=True, size=12); p.paragraph_format.space_before = Pt(10)
            p = doc.add_paragraph(); add_text(p, b[1]); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "keywords":
            p = doc.add_paragraph(); add_text(p, "Keywords: ", bold=True); add_text(p, b[1], italic=True)
        elif kind == "h1":
            p = doc.add_paragraph(); add_text(p, b[1], bold=True, size=13); p.paragraph_format.space_before = Pt(14); p.paragraph_format.keep_with_next = True
        elif kind == "h2":
            p = doc.add_paragraph(); add_text(p, b[1], bold=True, size=11.5); p.paragraph_format.space_before = Pt(8); p.paragraph_format.keep_with_next = True
        elif kind == "p":
            p = doc.add_paragraph(); add_text(p, b[1]); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "pb":
            p = doc.add_paragraph(); add_text(p, b[1] + " ", bold=True); add_text(p, b[2]); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "note":
            p = doc.add_paragraph(); add_text(p, b[1], italic=True, size=9.5)
        elif kind == "bullets":
            for t in b[1]:
                p = doc.add_paragraph(style="List Bullet"); add_text(p, t); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "eq":
            eq_n += 1
            p = doc.add_paragraph()
            ts = p.paragraph_format.tab_stops
            ts.add_tab_stop(Inches(3.25), WD_TAB_ALIGNMENT.CENTER)
            ts.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
            p.add_run("	")
            p._p.append(OM.tex_to_omml(b[1]))
            p.add_run("	(%d)" % eq_n)
            p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(8)
        elif kind == "fig":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(os.path.join(fig_dir, b[1]), width=Inches(b[3] if len(b) > 3 else 6.3)); p.paragraph_format.keep_with_next = True
            c = doc.add_paragraph(); add_text(c, b[2], italic=True, size=9.5)
        elif kind == "table":
            cap, header, rows = b[1], b[2], b[3]
            c = doc.add_paragraph(); add_text(c, cap, italic=True, size=9.5); c.paragraph_format.keep_with_next = True
            t = doc.add_table(rows=1 + len(rows), cols=len(header))
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            _borders(t)
            for j, h in enumerate(header):
                cell = t.rows[0].cells[j]; cell.text = ""
                add_text(cell.paragraphs[0], h, bold=True, size=8.5); _shade(cell, "E4ECF3")
            for i, row in enumerate(rows, 1):
                for j, v in enumerate(row):
                    cell = t.rows[i].cells[j]; cell.text = ""
                    add_text(cell.paragraphs[0], str(v), size=8.5)
            for row in t.rows:
                for cell in row.cells:
                    for pp in cell.paragraphs:
                        pp.paragraph_format.space_after = Pt(1); pp.paragraph_format.line_spacing = 1.0
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif kind == "ref":
            p = doc.add_paragraph(); add_text(p, b[1], size=9.5); p.paragraph_format.left_indent = Inches(0.3); p.paragraph_format.first_line_indent = Inches(-0.3)
            p.paragraph_format.space_after = Pt(2)
    doc.save(path)


# ----------------------------------------------------------------------------------------------------------- LaTeX
def _tex_escape_text(s):
    """Escape text outside math."""
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("_", "\\_")
             .replace("±", "$\\pm$").replace("×", "$\\times$").replace("−", "$-$").replace("≈", "$\\approx$").replace("≳", "$\\gtrsim$")
             .replace("≲", "$\\lesssim$").replace("→", "$\\to$").replace("≤", "$\\le$").replace("≥", "$\\ge$").replace("°", "$^\\circ$")
             .replace("–", "--").replace("—", "---").replace("’", "'").replace("“", "``").replace("”", "''").replace("µ", "$\\mu$")
             .replace("Λ", "$\\Lambda$").replace("λ", "$\\lambda$").replace("η", "$\\eta$").replace("ρ", "$\\rho$").replace("σ", "$\\sigma$").replace("φ", "$\\varphi$"))


def tex_inline(text):
    parts = re.split(r"(\$[^$]+\$)", text)
    out = []
    for p in parts:
        if p.startswith("$") and p.endswith("$") and len(p) > 1:
            out.append(p)
        else:
            out.append(_tex_escape_text(p))
    return "".join(out)


def render_tex(blocks, path, fig_dir_rel="../figures"):
    L = [r"\documentclass[11pt,a4paper]{article}", r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}", r"\usepackage{amsmath,amssymb}",
         r"\usepackage{graphicx,booktabs,array,longtable}", r"\usepackage[margin=2.5cm]{geometry}", r"\usepackage{hyperref}", r"\usepackage{setspace}",
         r"\onehalfspacing", r"\setlength{\parskip}{0.4em}", r"\setlength{\parindent}{0pt}", ""]
    fignum, tabnum = 0, 0
    body = []
    title, author = "", ""
    for b in blocks:
        k = b[0]
        if k == "title":
            title = tex_inline(b[1])
        elif k == "author":
            author = r" \\ ".join(tex_inline(x) for x in b[1].split("\n"))
        elif k == "abstract":
            body.append(r"\begin{abstract}" + "\n" + tex_inline(b[1]) + "\n" + r"\end{abstract}")
        elif k == "keywords":
            body.append(r"\noindent\textbf{Keywords:} \emph{" + tex_inline(b[1]) + "}\n")
        elif k == "h1":
            txt = re.sub(r"^\d+\.\s*", "", b[1])
            star = "*" if b[1].startswith(("Declarations", "References", "Acknowledg", "Author", "Data", "Code", "Funding", "Conflict", "Ethics", "Use of")) else ""
            body.append(r"\section%s{%s}" % (star, tex_inline(txt)))
        elif k == "h2":
            txt = re.sub(r"^\d+\.\d+\s*", "", b[1])
            body.append(r"\subsection{%s}" % tex_inline(txt))
        elif k == "p":
            body.append(tex_inline(b[1]) + "\n")
        elif k == "pb":
            body.append(r"\noindent\textbf{" + tex_inline(b[1]) + "} " + tex_inline(b[2]) + "\n")
        elif k == "note":
            body.append(r"\emph{" + tex_inline(b[1]) + "}\n")
        elif k == "bullets":
            body.append(r"\begin{itemize}" + "".join(r"\item " + tex_inline(t) + "\n" for t in b[1]) + r"\end{itemize}")
        elif k == "eq":
            body.append(r"\begin{equation}" + "\n" + b[1] + "\n" + r"\end{equation}")
        elif k == "fig":
            fignum += 1
            wfrac = min(1.0, (b[3] if len(b) > 3 else 6.3) / 6.3)
            body.append(r"\begin{figure}[htbp]\centering\includegraphics[width=%.2f\textwidth]{%s/%s}\caption{%s}\end{figure}" % (wfrac, fig_dir_rel, b[1], tex_inline(re.sub(r"^Fig\.\s*\d+\.\s*", "", b[2]))))
        elif k == "table":
            tabnum += 1
            header, rows = b[2], b[3]
            n_c = len(header)
            plain = lambda t: re.sub(r"[$\\{}^_]", "", str(t))
            lens = [min(60, max([len(plain(header[j]))] + [len(plain(r_[j])) for r_ in rows])) for j in range(n_c)]
            wide = max(sum(min(60, len(plain(r_[j]))) for j in range(n_c)) for r_ in rows) > 95
            if wide:
                tot = float(sum(lens))
                fr = [max(0.09, l_ / tot) for l_ in lens]
                z = sum(fr)
                col = "".join(r">{\raggedright\arraybackslash}p{%.3f\dimexpr\textwidth-%d\tabcolsep\relax}" % (f_ / z, 2 * n_c) for f_ in fr)
            else:
                col = "l" * n_c
            tb = [r"\begin{table}[htbp]\centering\small", r"\caption{%s}" % tex_inline(re.sub(r"^Table\s*\d+\.\s*", "", b[1])), r"\begin{tabular}{%s}" % col, r"\toprule",
                  " & ".join(tex_inline(h) for h in header) + r" \\ \midrule"]
            for r_ in rows:
                tb.append(" & ".join(tex_inline(str(v)) for v in r_) + r" \\")
            tb += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
            body.append("\n".join(tb))
        elif k == "ref":
            body.append(r"\noindent\hangindent=1.5em " + tex_inline(b[1]) + r"\par")
    L += [r"\title{" + title + "}", r"\author{" + author + "}", r"\date{}", r"\begin{document}", r"\maketitle", ""]
    L += body
    L += [r"\end{document}"]
    open(path, "w", encoding="utf-8").write("\n".join(L))
