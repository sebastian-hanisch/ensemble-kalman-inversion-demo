"""Ensemble-Kalman-Inversion: ableitungsfrei, aber mit großem Ensemble - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 9 (Anfangswert einer Wärmeleitung mit logistischer Reaktion) schätzt ein Ensemble die
Kovarianzen, die sonst die Jacobi-Matrix liefert: Update, getemperte Inversion und Iteration. Maßstab sind der exakte Posterior (linearer Fall), die Laplace-Näherung und MCMC
aus Stück 8. Kern: eki_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import eki_constants as C
from eki_evaluation import (
    Settings,
    analyse,
    count_text,
    de,
    err_pct,
    localization_experiment,
    loc_label,
    noise_label,
    num_text,
    rho_label,
    sci,
    size_experiment,
    solver_experiment,
    steps_experiment,
    verdict_instance,
    verdict_localization,
    verdict_size,
    verdict_solver,
    verdict_steps,
)
from eki_pdf_export import build_pdf
from eki_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from eki_visualization import (
    build_history_chart,
    build_loc_chart,
    build_recon_chart,
    build_sd_chart,
    build_size_chart,
    build_size_error_chart,
    build_solver_chart,
    build_steps_chart,
    build_table,
    style_table,
)

st.set_page_config(page_title="Ensemble-Kalman-Inversion – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _size():
    return size_experiment()


@st.cache_data(show_spinner=SPINNER)
def _loc():
    return localization_experiment()


@st.cache_data(show_spinner=SPINNER)
def _steps():
    return steps_experiment()


@st.cache_data(show_spinner=SPINNER)
def _solver():
    return solver_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎲 Ensemble-Kalman-Inversion: Wie groß muss das Ensemble sein?")
st.markdown(
    """
**Die Ensemble-Kalman-Inversion braucht keine Ableitung, aber ein großes Ensemble und im nichtlinearen Fall mehrere Schritte.** Dies ist das zehnte Stück der Linie
**Inverse Probleme und Regularisierung** und baut auf **Nichtlineare Probleme** und **Posterior-Sampling** auf: dasselbe Phantom, dasselbe Modell (Anfangswert einer Wärmeleitung mit
logistischer Reaktion), aber statt der Jacobi-Matrix schätzt ein **Ensemble** von Kandidaten die Kovarianzen, die das Kalman-Update braucht. Im linear-gaußschen Fall ist der exakte Posterior
bekannt und der Maßstab; sonst sind es die Laplace-Näherung und MCMC aus Stück 8. Die Frage ist, **wie viele Mitglieder** nötig sind, **wie viele Schritte** das nichtlineare Problem
braucht, was eine **Lokalisierung** rettet und wie sich das Verfahren als **ableitungsfreier Löser** gegen Gauss-Newton schlägt. Die Welt ist simuliert; die Wahrheit ist bekannt und dient nur als Maßstab.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie ein Ensemble-Kalman-Schritt funktioniert", expanded=True):
    st.markdown(
        r"""
1. **Ensemble.** $J$ Kandidaten $x_j$ werden aus dem Prior gezogen (exponentieller Kern wie in Stück 8) und durch das Modell geschickt: $g_j = G(x_j)$.
2. **Kovarianzen aus dem Ensemble.** $C_{xg}$ (Zustand mit Beobachtung) und $C_{gg}$ (Beobachtung mit Beobachtung) sind die Stichproben-Kovarianzen der $J$ Mitglieder. Es braucht keine Ableitung von $G$.
3. **Update.** Jedes Mitglied wird verschoben: $x_j \leftarrow x_j + C_{xg}\,(C_{gg} + \alpha\Gamma)^{-1}\,(y + \sqrt\alpha\,\eta_j - g_j)$ mit Störung $\eta_j \sim \mathcal N(0, \Gamma)$, $\Gamma = \sigma^2 I$.
4. **Verfahren.** *Update*: ein Schritt mit $\alpha = 1$. *Tempering*: $N$ Schritte mit $\alpha = N$ (im linear-gaußschen Fall und für $J \to \infty$ der exakte Posterior). *Iteration*: viele Schritte mit $\alpha = 1$, Halt nach dem Diskrepanzprinzip (ein ableitungsfreier Löser).
5. **Lokalisierung.** Bei kleinem Ensemble werden $C_{xg}$ und $C_{gg}$ mit der Gaspari-Cohn-Funktion über den Abstand der Gitterpunkte gedämpft (beide, sonst passt der Kreuzterm nicht mehr zu $C_{gg}$).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt und Messung**")
    rho = st.select_slider("Reaktionsstärke ρ", options=C.RHO_CHOICES, key="rho_select", format_func=rho_label,
                           help="0 ist das lineare Modell (exakter Posterior als Referenz); je größer, desto nichtlinearer (Stück 9). Bei ρ > 0 ist die Referenz die Laplace-Näherung.")
    data = st.selectbox("Woher kommen die Daten?", C.DATA_KINDS, key="data_select", format_func=lambda v: C.DATA_LABELS[v],
                        help="Ehrlich: aus einem feineren Modell mit 32 Schritten. Inverse crime: aus demselben Modell, das invertiert wird.")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Wie viele Messpunkte und Unbekannte.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f", help="Breite der Wärmeleitung nach vier Schritten, als Anteil der Länge 1.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten; je kleiner, desto informativer die Daten und desto größer muss das Ensemble sein.")
    st.markdown("**Verfahren**")
    method = st.selectbox("Welches Verfahren?", C.METHODS, key="method_select", format_func=lambda v: C.METHOD_LABELS[v],
                          help="Update: ein Schritt. Tempering: N Schritte mit α = N, nähert den Posterior. Iteration: α = 1 mit Halt nach dem Diskrepanzprinzip, ein Löser ohne Streuungsmaß.")
    j = st.select_slider("Ensemblegröße J", options=C.J_CHOICES, key="j_select", format_func=count_text,
                         help="Zahl der Mitglieder. Zu klein: die Streuung bricht zusammen. Es zählt die Zahl der von den Daten bestimmten Richtungen, nicht allein n.")
    steps = st.select_slider("Schritte N", options=C.STEP_CHOICES, key="steps_select", help="Tempering: Zahl der Schritte (α = N). Iteration: höchste Zahl der Iterationen. Beim Update ohne Wirkung.")
    loc = st.select_slider("Lokalisierung (Halbbreite)", options=C.LOC_CHOICES, key="loc_select", format_func=loc_label,
                           help="Gaspari-Cohn-Halbbreite als Anteil der Länge 1 (Träger = 2 × Halbbreite); aus = 0. Zu klein macht die Unsicherheit zu groß.")
    seed = st.number_input("Zufalls-Seed des Rauschens und des Ensembles", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "rho_select": float(rho), "data_select": data, "method_select": method,
                   "j_select": int(j), "steps_select": int(steps), "loc_select": float(loc), "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), float(rho), data, method, int(j), int(steps), float(loc), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.diverged:
    st.info("Das Ensemble ist davongelaufen (Fehler über dem Dreifachen der Signalnorm oder nichtendliche Werte); gezeigt ist das letzte endliche Ensemble, das Bild kann außerhalb des Achsenbereichs liegen.")
c1, c2, c3, c4 = st.columns(4)
if method == "iteration":
    c1.metric("Fehler des Ensemblemittels", err_pct(a.err), delta=f"MAP: {err_pct(a.err_map)}", delta_color="off", help="Abstand zur Wahrheit relativ zur Signalnorm; das MAP (Levenberg-Marquardt, mit Jacobi-Matrix) ist der Maßstab.")
    c2.metric("Halt (Diskrepanzprinzip)", "–" if a.stop is None else f"Iteration {a.stop}", delta=f"Residuum {num_text(a.resid[-1])}-fach", delta_color="off",
              help="Erste Iteration, nach der |G(Mittel) − b| höchstens das 1,5-Fache von σ√m beträgt; am Ende der Läufe das Residuum relativ zum Rauschniveau.")
    c3.metric("Streuung des Ensembles", f"{de(100.0 * float(a.sd.mean()) / a.prior_spread, 1)} %", delta="des Priors", delta_color="off", help="Mittlere Standardabweichung der Komponenten relativ zum Prior; nahe 0 heißt zusammengebrochen.")
else:
    c1.metric("Mittel gegen Referenz", f"{de(a.mean_err_sd, 2)} sd", delta="Soll: unter 0,5", delta_color="off", help=f"Abstand des Ensemblemittels zum Referenzmittel ({a.ref_kind}) in dessen Standardabweichungen (RMS über die Komponenten).")
    c2.metric("sd-Verhältnis", de(a.sd_ratio, 2), delta=f"{de(a.sd_ratio_min, 2)} bis {de(a.sd_ratio_max, 2)}", delta_color="off", help="Standardabweichung des Ensembles geteilt durch die der Referenz, Median über die Komponenten; Soll 0,8 bis 1,25.")
    c3.metric("Fehler gegen die Wahrheit", err_pct(a.err), delta=f"Referenz: {err_pct(a.err_ref)}", delta_color="off", help="Ensemblemittel gegen Wahrheit, relativ zur Signalnorm.")
c4.metric("Vorwärtsläufe", count_text(a.evals), delta=f"{a.steps_run} {'Schritt' if a.steps_run == 1 else 'Schritte'} × J", delta_color="off",
          help="Aufwand: J Läufe je Schritt. Das MAP braucht dagegen nur eine Jacobi-Matrix je Iteration (und die Ableitung des Modells).")
st.caption(f"Referenz: {a.ref_kind}; MAP {err_pct(a.err_map)} nach {a.map_iters} Iterationen mit {a.map_jacs} Jacobi-Matrizen; mittlere Streuung des Ensembles {sci(float(a.sd.mean()))} (Prior {de(a.prior_spread, 2)}).")

tab_rec, tab_sd, tab_hist, tab_tab = st.tabs(["Rekonstruktion", "Unsicherheit", "Verlauf", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, blau das Ensemblemittel mit ±2 sd aus dem Ensemble, schwarz punktiert die Referenz (±2 sd), grau einige Mitglieder. Die Achse ist fest.")
with tab_sd:
    st.plotly_chart(build_sd_chart(a), width="stretch", key="sd_chart")
    st.caption("Standardabweichung des Ensembles gegen die Referenz je Gitterpunkt (logarithmisch).")
with tab_hist:
    st.plotly_chart(build_history_chart(a), width="stretch", key="history_chart")
    st.caption("Fehler des Ensemblemittels, mittlere Streuung und Residuum (relativ zum Rauschniveau) nach jedem Schritt. Ein Ensemble, dessen Streuung auf null fällt, sieht „fertig“ aus, ohne es zu sein.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="ensemble-kalman-inversion.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist die Unsicherheit des Ensembles brauchbar?")
verdict(verdict_instance(a))
st.caption("Das Urteil vergleicht Mittel und Streuung des Ensembles mit der Referenz (beim Update und beim Tempering) beziehungsweise den Fehler mit dem des MAP und den Halt nach dem Diskrepanzprinzip (bei der Iteration).")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie groß muss das Ensemble sein?")
res1 = _size()
st.caption(f"Linear (ρ = 0), ein Update gegen den exakten Posterior, ehrliche Daten, {res1['reps']} Wiederholungen je Zelle (Median); Teil 1 bei n = {res1['n']} über die Ensemblegröße und drei Rauschstufen, "
           "Teil 2 über die Dimension n.")
cs1, cs2 = st.columns(2)
with cs1:
    st.plotly_chart(build_size_chart(res1), width="stretch", key="size_chart")
with cs2:
    st.plotly_chart(build_size_error_chart(res1), width="stretch", key="size_error_chart")
verdict(verdict_size(res1))
st.dataframe(
    [{"Rauschen": noise_label(r["noise"]), "J": count_text(r["j"]), "Mittelwert (sd)": de(r["dm"], 2), "sd-Verhältnis": de(r["ratio"], 2), "min bis max": f"{de(r['ratio_min'], 2)} bis {de(r['ratio_max'], 2)}",
      "Fehler des Mittels": err_pct(r["err"]), "Fehler exakt": err_pct(r["err_exact"]), "bestimmte Richtungen": r["directions"]} for r in res1["rows"]], hide_index=True, width="stretch", height=360)
st.dataframe([{"n": d["n"], "J": d["j"], "Mittelwert (sd)": de(d["dm"], 2), "sd-Verhältnis": de(d["ratio"], 2), "bestimmte Richtungen": d["directions"]} for d in res1["dims"]], hide_index=True, width="stretch")
st.caption("Der Mittelwertfehler fällt wie 1/√J; die Streuung ist erst bei einem Vielfachen der von den Daten bestimmten Richtungen getroffen. Maßgeblich ist diese Zahl (Richtungen mit prior-gewichtetem Hesse-Eigenwert über 1, Stück 8), "
           "nicht n allein: bei n = 24 reicht J = 50, bei n = 48 und 96 nicht.")

st.subheader("🔬 Experiment 2: Was rettet ein kleines Ensemble?")
res2 = _loc()
st.caption(f"Linear, Rauschen {noise_label(C.LOC_NOISE)}, ein Update mit Gaspari-Cohn-Lokalisierung von C_xg und C_gg, {res2['reps']} Wiederholungen je Zelle, nach Ensemblegröße und Halbbreite.")
st.plotly_chart(build_loc_chart(res2), width="stretch", key="loc_chart")
verdict(verdict_localization(res2))
st.dataframe([{"J": r["j"], "Halbbreite": loc_label(r["c"]), "Mittelwert (sd)": de(r["dm"], 2), "sd-Verhältnis": de(r["ratio"], 2), "min bis max": f"{de(r['ratio_min'], 2)} bis {de(r['ratio_max'], 2)}"}
              for r in res2["rows"]], hide_index=True, width="stretch", height=360)
st.caption("Lokalisierung ersetzt Mitglieder durch Vorwissen: dass nur nahe Messungen etwas über einen Punkt sagen. Eine zu kleine Halbbreite wirft zu viel davon weg und macht die Unsicherheit zu groß; "
           "gedämpft werden müssen beide Kovarianzen, sonst passt der Kreuzterm nicht mehr zu C_gg.")

st.subheader("🔬 Experiment 3: Wie viele Schritte braucht die nichtlineare Welt?")
res3 = _steps()
st.caption(f"Ensemble mit J = {C.STEPS_J} und N Schritten (α = N) gegen MCMC ({res3['chains']} Ketten à {res3['steps']} Schritte, Zufallsschritt-Metropolis mit der Gauss-Newton-Hesse-Matrix, Rauschen "
           f"{noise_label(C.STEPS_NOISE)}), {C.STEPS_REPS} Wiederholungen; unten der lineare Fall gegen den exakten Posterior.")
st.plotly_chart(build_steps_chart(res3), width="stretch", key="steps_chart")
verdict(verdict_steps(res3))
st.dataframe(
    [{"ρ": rho_label(r["rho"]), "Schritte": n_, "Mittelwert gegen MCMC (sd)": de(v["dm"], 2), "sd-Verhältnis": de(v["ratio"], 2), "min bis max": f"{de(v['ratio_min'], 2)} bis {de(v['ratio_max'], 2)}",
      "Fehler des Ensemblemittels": err_pct(v["err"]), "Fehler MCMC-Mittel": err_pct(r["err_mcmc"]), "Laplace / MCMC (sd)": de(r["laplace_ratio"], 2), "R̂ MCMC": de(r["rhat"], 2)}
     for r in res3["rows"] for n_, v in r["by_n"].items()], hide_index=True, width="stretch", height=360)
st.dataframe([{"J": d["j"], "Schritte": d["n"], "Mittelwert (sd)": de(d["dm"], 2), "sd-Verhältnis": de(d["ratio"], 2)} for d in res3["linear"]], hide_index=True, width="stretch")
st.caption("Das einzelne Update behandelt die Abbildung als linear und ist in der nichtlinearen Welt zu breit; mehrere gleiche Schritte treffen das Mittel, schrumpfen aber die Streuung. Im linearen Fall bleibt das Mittel, "
           "und jeder weitere Schritt kostet etwas Streuung (Stichprobenfehler der Kovarianzen summiert sich).")

st.subheader("🔬 Experiment 4: Ableitungsfreier Löser")
res4 = _solver()
st.caption(f"α = 1 iteriert (Prior nur im Start-Ensemble), {C.SOLVER_ITERS} Iterationen, Rauschen {noise_label(C.SOLVER_NOISE)}, ehrliche Daten, {res4['reps']} Wiederholungen je Zelle (Median); "
           "der Halt nach dem Diskrepanzprinzip und das MAP (Levenberg-Marquardt mit Jacobi-Matrix) als Vergleich.")
cv1, cv2 = st.columns(2)
with cv1:
    st.plotly_chart(build_solver_chart(res4, 1.0), width="stretch", key="solver_chart_1")
with cv2:
    st.plotly_chart(build_solver_chart(res4, 2.0), width="stretch", key="solver_chart_2")
verdict(verdict_solver(res4))
st.dataframe(
    [{"ρ": rho_label(r["rho"]), "J": r["j"], "Halt gefunden": f"{r['stop_found']} von {res4['reps']}", "Halt (Median)": "–" if r["stop"] is None else de(r["stop"], 1), "Fehler am Halt": err_pct(r["err_stop"]),
      "Fehler nach 30": err_pct(r["err_final"]), "davongelaufen": f"{r['diverged']} von {res4['reps']}", "Fehler MAP": err_pct(r["err_map"]), "Jacobi-Matrizen MAP": r["map_jacs"]}
     for r in res4["rows"]], hide_index=True, width="stretch")
st.caption("Mit zu kleinem Ensemble bricht die Streuung nach wenigen Iterationen zusammen und das Mittel bleibt stehen; in der nichtlinearen Welt kann es sogar explodieren. Mit 200 Mitgliedern erreicht das Verfahren das "
           "Rauschniveau nach wenigen Iterationen, bleibt aber über dem Fehler des MAP und überanpasst danach langsam: wie in Stück 5 ist der Abbruch die Regularisierung.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
rows1 = {(r["noise"], r["j"]): r for r in res1["rows"]}
by3 = {r["rho"]: r["by_n"] for r in res3["rows"]}
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ensemble ist groß genug** | Bei Rauschen 1 % und J = 25 liegt das sd-Verhältnis bei {de(rows1[(1e-2, 25)]['ratio'], 2)}, bei J = 10 bei {de(rows1[(1e-2, 10)]['ratio'], 2)}; das Mittel liegt {de(rows1[(1e-2, 25)]['dm'], 1)} sd daneben. | Lokalisierung (Experiment 2) |
| **Die Abbildung ist nahezu linear** | Bei ρ = 1 ist das einzelne Update zu breit (sd-Verhältnis {de(by3[1.0][1]['ratio'], 2)} gegen MCMC). | Stück 9 (Gauss-Newton) |
| **Der Posterior ist gaußsch** | Auch viele getemperte Schritte unterschätzen die Streuung (ρ = 4, 16 Schritte: {de(by3[4.0][16]['ratio'], 2)}); die Näherung kennt keine Schiefe. | Stück 15 (Diffusions-Prior) |
| **Die Referenz stimmt** | Bei ρ > 0 ist die Referenz in der Instanzansicht die Laplace-Näherung; sie stimmt mit MCMC bei ρ ≤ 2 auf 1 % überein, bei ρ = 4 weniger. | – |
| **Ein Modell, ein Prior, ein Gitter** | Wärmeleitung und logistische Reaktion (Stück 9), exponentieller Prior, eine Dimension; die Lokalisierung nutzt, dass der Operator lokal ist. | – |
| **Störbeobachtungen** | Ein Wurzel-Filter (ETKF) würde die Stichproben-Störung der Beobachtungen sparen; nicht gebaut. | – |
| **Der Löser braucht kein Streuungsmaß** | Im Löser-Modus gibt es keinen Posterior; ein zusammengebrochenes Ensemble sieht fertig aus. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $b = G(x) + e$, $e \sim \mathcal N(0, \sigma^2 I)$, $G$ wie in Stück 9 (Wärmeleitung und logistische Reaktion, $K = 4$ Schritte); Prior $x \sim \mathcal N(0, \tau^2 C)$ mit $C_{ij} = e^{-|s_i - s_j|/\ell}$.

**Ensemble-Kalman-Schritt.** Mit Mitgliedern $x_j$, $g_j = G(x_j)$, Anomalien $A = (x_j - \bar x)_j$, $B = (g_j - \bar g)_j$ und den Stichproben-Kovarianzen $C_{xg} = A^\top B/(J-1)$, $C_{gg} = B^\top B/(J-1)$:
$x_j \leftarrow x_j + C_{xg}(C_{gg} + \alpha\sigma^2 I)^{-1}(y + \sqrt\alpha\,\eta_j - g_j)$, $\eta_j \sim \mathcal N(0, \sigma^2 I)$.

**Posterior im linearen Fall.** Mit $G(x) = Ax$ und den exakten Kovarianzen ist ein Schritt mit $\alpha = 1$ die Kalman-Formel: Mittel $m = \tau^2 C A^\top(\tau^2 A C A^\top + \sigma^2 I)^{-1} b$, Kovarianz $\Gamma = \tau^2 C - \tau^2 C A^\top(\tau^2 A C A^\top + \sigma^2 I)^{-1}\tau^2 A C$ (Stück 7).
$N$ Schritte mit $\alpha = N$ und exakten Kovarianzen liefern dasselbe, weil die Genauigkeiten der $N$ Beobachtungen $1/(N\sigma^2)$ sich zu $1/\sigma^2$ addieren. Mit Stichproben-Kovarianzen ist es nur im Grenzwert $J \to \infty$ so (Fehler $\sim 1/\sqrt J$).

**Teilraum-Eigenschaft.** Der Zuwachs liegt im Aufspann der $J$ Anomalien: Bei $J \le n$ kann das Ensemble nur Richtungen korrigieren, die schon im Start-Ensemble vorkommen.

**Lokalisierung.** $C_{xg} \leftarrow C_{xg} \circ T$, $C_{gg} \leftarrow C_{gg} \circ T$ mit der Gaspari-Cohn-Funktion $T_{ij} = \rho_{\mathrm{GC}}(|s_i - s_j|/c)$ (Träger $2c$); das Schur-Produkt mit einer positiv semidefiniten Matrix lässt $C_{gg}$ positiv semidefinit.

**Laplace-Näherung und MCMC.** $\pi(x \mid b) \approx \mathcal N(x_{\mathrm{MAP}}, H^{-1})$ mit $H = J^\top J/\sigma^2 + (\tau^2 C)^{-1}$ am MAP (Stück 9); der Sampler ist das vorkonditionierte Zufallsschritt-Metropolis-Verfahren aus Stück 8.

**Literatur.** Das Beispiel ist erfunden; die Ensemble-Kalman-Inversion geht auf M. A. Iglesias, K. J. H. Law und A. M. Stuart (2013) zurück, das Tempering (Mehrfachzuordnung der Daten) auf A. A. Emerick und A. C. Reynolds (2013),
die Lokalisierung auf G. Gaspari und S. E. Cohn (1999), das Diskrepanzprinzip für den Löser-Modus auf M. A. Iglesias (2016) (alles aus dem Gedächtnis, nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html)."
)
