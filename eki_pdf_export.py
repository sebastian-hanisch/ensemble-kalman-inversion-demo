"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import eki_constants as C
from eki_evaluation import Analysis, count_text, de, err_pct, loc_label, num_text, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "ρ": "rho", "α": "alpha", "±": "+-", "√": "sqrt", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "⁻": "-", "⁰": "0",
            "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Ensemble-Kalman-Inversion"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Gitterpunkte n = {s.n}, Breite der Unschärfe w = {de(s.w, 3)}, Rauschen {de(100 * s.noise, 4)} % des Effektivwerts, Reaktionsstärke rho = {de(s.rho, 1)}, "
                            f"Daten: {C.DATA_SHORT[s.data]}, Verfahren: {C.METHOD_SHORT[s.method]}, Ensemblegröße J = {s.j}, {a.steps_run} Schritte (alpha = {de(a.alpha, 0)}), "
                            f"Lokalisierung: {loc_label(s.loc)}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Vorwärtsläufe: {count_text(a.evals)}; Referenz: {a.ref_kind}; Mittelwertabstand zur Referenz {de(a.mean_err_sd, 2)} sd; "
        f"sd-Verhältnis (Median / kleinster / größter Wert) {de(a.sd_ratio, 2)} / {de(a.sd_ratio_min, 2)} / {de(a.sd_ratio_max, 2)}.",
        f"Fehler des Ensemblemittels {err_pct(a.err)}; Referenzmittel {err_pct(a.err_ref)}; MAP {err_pct(a.err_map)} ({a.map_iters} Iterationen, {a.map_jacs} Jacobi-Matrizen).",
        (f"Halt nach dem Diskrepanzprinzip: Iteration {a.stop}." if a.stop is not None else "Halt nach dem Diskrepanzprinzip: nicht erreicht oder nicht verlangt.")
        + f" Residuum / (sigma sqrt m) am Ende: {num_text(a.resid[-1])}; mittlere Streuung {sci(a.spreads[-1])} (Prior: {de(a.prior_spread, 2)}).",
        f"Wahres Rauschen sigma = {sci(a.sigma)}; Ensemble " + ("endlich." if a.finite else "mit nichtendlichen Werten abgebrochen (letztes endliches Ensemble gezeigt)."),
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "Mittel", "Ensemble-sd", "Referenz"], [16, 26, 32, 34, 36, 36]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(s.n, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), num_text(float(a.mean[i])), num_text(float(a.sd[i])), de(a.ref_mean[i], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if s.n > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {s.n - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Ensemble-Kalman-Inversion für nichtlineare Inverse Probleme. Quelle: ensemble-kalman-inversion-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
