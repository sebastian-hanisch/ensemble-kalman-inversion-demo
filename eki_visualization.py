"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import eki_constants as C
from eki_evaluation import Analysis, de, loc_label, noise_label, rho_label


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({"Punkt": [f"{i + 1}" for i in range(k)], "Ort s": a.s[:k], "Wahrheit x": a.x[:k], "Ensemblemittel": a.mean[:k], "Ensemble-sd": a.sd[:k],
                         f"Referenz-Mittel ({a.ref_kind})": a.ref_mean[:k], f"Referenz-sd ({a.ref_kind})": a.ref_sd[:k], "Daten b": a.b[:k]})


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 4)) for c in df.columns if c != "Punkt"})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, Ensemblemittel mit ±2 sd aus dem Ensemble, Referenz (±2 sd) und einige Mitglieder; die Achse ist fest."""
    lo, hi = C.SHOW_RANGE
    fig = go.Figure()
    for k, member in enumerate(a.members):
        fig.add_trace(go.Scatter(x=a.s, y=member, mode="lines", line=dict(color="rgba(138,138,138,0.35)", width=1), name="Mitglieder", legendgroup="members", showlegend=k == 0, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.mean + C.Z_BAND * a.sd, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.mean - C.Z_BAND * a.sd, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="Ensemble ±2 sd", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.ref_mean + C.Z_BAND * a.ref_sd, mode="lines", name=f"Referenz ({a.ref_kind}) ±2 sd", line=dict(color=C.COLOR_REF, width=1, dash="dot"), legendgroup="ref",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.ref_mean - C.Z_BAND * a.ref_sd, mode="lines", line=dict(color=C.COLOR_REF, width=1, dash="dot"), legendgroup="ref", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.mean, mode="lines+markers", name="Ensemblemittel", line=dict(color=C.COLOR_ENSEMBLE, width=2.5), marker=dict(size=5),
                             hovertemplate="s %{x:.3f}<br>Mittel %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    return _fixed(fig, 400)


def build_sd_chart(a: Analysis) -> go.Figure:
    """Standardabweichung des Ensembles gegen die Referenz je Gitterpunkt (logarithmisch)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=np.maximum(a.sd, 1e-12), mode="lines+markers", name="Ensemble", line=dict(color=C.COLOR_ENSEMBLE, width=3), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=a.s, y=a.ref_sd, mode="lines", name=f"Referenz ({a.ref_kind})", line=dict(color=C.COLOR_REF, width=2, dash="dash")))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Standardabweichung", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_history_chart(a: Analysis) -> go.Figure:
    """Fehler des Ensemblemittels, mittlere Streuung und Residuum über die Schritte (logarithmisch); der Halt nach dem Diskrepanzprinzip ist markiert."""
    k = list(range(len(a.errs)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=np.maximum(a.errs, 1e-12), mode="lines+markers", name="Fehler des Mittels (relativ)", line=dict(color=C.COLOR_ENSEMBLE, width=3), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=k, y=np.maximum(a.spreads, 1e-12), mode="lines+markers", name="mittlere Streuung des Ensembles", line=dict(color=C.COLOR_MAP, width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=k, y=np.maximum(a.resid, 1e-12), mode="lines+markers", name="Residuum / (σ√m)", line=dict(color=C.COLOR_GREY, width=2, dash="dash"), marker=dict(size=5)))
    if a.stop is not None:
        fig.add_vline(x=a.stop, line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), annotation_text="Halt", annotation_position="top right")
    fig.add_hline(y=C.DISC_FACTOR, line=dict(color=C.COLOR_GREY, width=1, dash="dot"))
    fig.update_xaxes(title="Schritt")
    fig.update_yaxes(title="Wert (logarithmisch)", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_size_chart(res: dict) -> go.Figure:
    """sd-Verhältnis (Median) über die Ensemblegröße, je Rauschstufe."""
    fig = go.Figure()
    for noise, color in zip(C.SIZE_NOISE, C.COLOR_J):
        rows = [r for r in res["rows"] if r["noise"] == noise]
        fig.add_trace(go.Scatter(x=[r["j"] for r in rows], y=[r["ratio"] for r in rows], mode="lines+markers", name=f"Rauschen {noise_label(noise)}", line=dict(color=color, width=3), marker=dict(size=8)))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="Ensemble-sd = exakte sd", annotation_position="bottom right")
    fig.update_xaxes(title="Ensemblegröße J", type="log", tickvals=list(C.SIZE_J), ticktext=[str(j) for j in C.SIZE_J])
    fig.update_yaxes(title="sd-Verhältnis (Median über die Komponenten)", range=[-0.02, 1.15])
    return _fixed(fig, 340)


def build_size_error_chart(res: dict) -> go.Figure:
    """Mittelwertabstand zum exakten Posterior in sd über die Ensemblegröße (logarithmisch), mit Streubereich über die Wiederholungen."""
    fig = go.Figure()
    for noise, color in zip(C.SIZE_NOISE, C.COLOR_J):
        rows = [r for r in res["rows"] if r["noise"] == noise]
        fig.add_trace(go.Scatter(x=[r["j"] for r in rows], y=[r["dm"] for r in rows], mode="lines+markers", name=f"Rauschen {noise_label(noise)}", line=dict(color=color, width=3), marker=dict(size=8),
                                 error_y=dict(type="data", symmetric=False, array=[r["dm_max"] - r["dm"] for r in rows], arrayminus=[r["dm"] - r["dm_min"] for r in rows], color=color, thickness=1)))
    fig.update_xaxes(title="Ensemblegröße J", type="log", tickvals=list(C.SIZE_J), ticktext=[str(j) for j in C.SIZE_J])
    fig.update_yaxes(title="Mittelwertabstand in sd (RMS; Balken: kleinster bis größter)", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_loc_chart(res: dict) -> go.Figure:
    """sd-Verhältnis (Median) über die Ensemblegröße, je Halbbreite der Lokalisierung."""
    fig = go.Figure()
    for c, color in zip(C.LOC_RADII, C.COLOR_J):
        rows = [r for r in res["rows"] if r["c"] == c]
        fig.add_trace(go.Scatter(x=[r["j"] for r in rows], y=[r["ratio"] for r in rows], mode="lines+markers", name=f"Halbbreite {loc_label(c)}", line=dict(color=color, width=3), marker=dict(size=8)))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="Ensemble-sd = exakte sd", annotation_position="bottom right")
    fig.update_xaxes(title="Ensemblegröße J", type="log", tickvals=list(C.LOC_J), ticktext=[str(j) for j in C.LOC_J])
    fig.update_yaxes(title="sd-Verhältnis (Median über die Komponenten)", range=[-0.02, 1.8])
    return _fixed(fig, 340)


def build_steps_chart(res: dict) -> go.Figure:
    """sd-Verhältnis (Median; Balken: kleinster bis größter Wert) Ensemble / MCMC je Reaktionsstärke und Schrittzahl."""
    fig = go.Figure()
    rhos = [r["rho"] for r in res["rows"]]
    for steps, color in zip(C.STEPS_N, C.COLOR_J[2:]):
        vals = [next(r for r in res["rows"] if r["rho"] == rho)["by_n"][steps] for rho in rhos]
        fig.add_trace(go.Bar(x=[rho_label(v) for v in rhos], y=[v["ratio"] for v in vals], name=f"{steps} {'Schritt' if steps == 1 else 'Schritte'}", marker_color=color,
                             error_y=dict(type="data", symmetric=False, array=[v["ratio_max"] - v["ratio"] for v in vals], arrayminus=[v["ratio"] - v["ratio_min"] for v in vals], color="#555555"),
                             text=[de(v["ratio"], 2) for v in vals], textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="Ensemble-sd = MCMC-sd", annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Reaktionsstärke ρ", type="category")
    fig.update_yaxes(title="sd-Verhältnis Ensemble / MCMC", range=[0, 5.2])
    return _fixed(fig, 360)


def build_solver_chart(res: dict, rho: float) -> go.Figure:
    """Fehler des Ensemblemittels (Median über die Wiederholungen) über die Iterationen, je Ensemblegröße; die Linie des MAP-Fehlers als Maßstab."""
    fig = go.Figure()
    rows = [r for r in res["rows"] if r["rho"] == rho]
    for r, color in zip(rows, C.COLOR_J[:3]):
        fig.add_trace(go.Scatter(x=list(range(len(r["errs"]))), y=[min(v, 1e4) for v in r["errs"]], mode="lines+markers", name=f"J = {r['j']}", line=dict(color=color, width=3), marker=dict(size=5)))
    fig.add_hline(y=rows[0]["err_map"], line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="Fehler des MAP", annotation_position="bottom right")
    fig.update_xaxes(title=f"Iteration (ρ = {rho_label(rho)})")
    fig.update_yaxes(title="Fehler des Ensemblemittels (relativ)", type="log", exponentformat="power")
    return _fixed(fig, 340)
