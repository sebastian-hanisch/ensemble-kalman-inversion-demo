# Ensemble-Kalman-Inversion: Wie groß muss das Ensemble sein? – Streamlit-Demo

Zehntes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [nichtlinear-inverse-demo](https://github.com/sebastian-hanisch/nichtlinear-inverse-demo), [posterior-sampling-demo](https://github.com/sebastian-hanisch/posterior-sampling-demo), [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo) und [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten als Anfangswert einer Wärmeleitung mit logistischer Reaktion, gemessen mit weißem Rauschen. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist; sie dient hier nur als Maßstab.

## Kernfrage

Stück 9 löste das nichtlineare Problem mit der Jacobi-Matrix. Hier schätzt ein **Ensemble** von Kandidaten die Kovarianzen, die das Kalman-Update braucht, ganz ohne Ableitung. Wie viele Mitglieder braucht es? Wie viele Schritte braucht die nichtlineare Welt? Was rettet ein kleines Ensemble? Und wie schlägt sich das Verfahren als **ableitungsfreier Löser** gegen das MAP aus Stück 9?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, Gitter, Rauschen):

- **Vorwärtsmodell** G aus Stück 9: u₀ = x, u_{k+1} = R(A u_k), G(x) = u_K mit K = 4 Schritten, A ein Wärmeleitungsschritt mit reflektierenden Rändern (Kosinusbasis), R(u) = u + (ρ/K) u (1 − u). Für ρ = 0 ist G linear. Standardinstanz n = 48, w = 0,03.
- **Daten:** ehrlich aus einem feineren Modell mit 32 Schritten (Standard) oder aus demselben Modell (inverse crime); Rauschen 1 % des Effektivwerts (Experiment 4: 0,1 %).
- **Prior:** exponentieller Kern (ℓ = 0,12, τ = 0,5) wie in Stück 8; das Start-Ensemble ist eine Ziehung daraus.
- **Verfahren** (je Schritt, mit den Stichproben-Kovarianzen C_xg, C_gg der J Mitglieder): x_j ← x_j + C_xg (C_gg + αΓ)⁻¹ (y + √α η_j − G(x_j)), η_j ~ N(0, Γ), Γ = σ² I. **Update:** ein Schritt, α = 1. **Tempering:** N Schritte mit α = N (im linear-gaußschen Fall und für J → ∞ der exakte Posterior). **Iteration:** α = 1 wiederholt, Halt nach dem Diskrepanzprinzip (Residuum höchstens das 1,5-Fache von σ√m), ein Löser ohne Posterior.
- **Lokalisierung:** Gaspari-Cohn-Funktion über den Abstand der Gitterpunkte, auf C_xg **und** C_gg.
- **Referenzen:** exakter Posterior (ρ = 0, Stück 7), Laplace-Näherung am MAP (Stück 9, gegen MCMC geprüft), MCMC aus Stück 8 (20 Ketten × 20.000 Schritte, vorkonditioniert mit der Gauss-Newton-Hesse-Matrix, überdispers gestartet, R̂ ≤ 1,06).

Alles ist erfunden; die Wahrheit ist das simulierte Phantom.

## Methodik

- **Kennzahlen:** Mittelwertabstand zur Referenz in Standardabweichungen der Referenz (RMS über die 48 Komponenten) und sd-Verhältnis Ensemble / Referenz (Median über die Komponenten).
- **Verdikt:** „zu wenig Streuung“ unter 0,8, „zu viel“ über 1,25, Mittelwertabstand über 0,5 sd; im Löser-Modus Halt erreicht und Fehler höchstens das Doppelte des MAP-Fehlers.
- **Experimente** laufen über feste Zufallsströme (SplitMix64); nie ein einzelner Seed. Jede Zelle ist der Median über 10 bis 30 Ensemble-Ziehungen.

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03), ehrliche Daten. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Wie groß muss das Ensemble sein (Experiment 1, linear, ein Update, Rauschen 1 %)? | J = 10 / 25 / 50 / 100 / 400 / 1.600 / 6.400: sd-Verhältnis **0,02 / 0,20 / 0,67 / 0,86 / 0,96 / 0,99 / 1,00**, Mittelwertabstand zum exakten Posterior **3,4 / 2,7 / 0,8 / 0,46 / 0,21 / 0,09 / 0,05 sd**: der Mittelwertfehler fällt wie 1/√J. Je informativer die Daten, desto größer das nötige Ensemble: Rauschen 10 % / 1 % / 0,1 % bei J = 50: 0,79 / 0,67 / 0,56. | `test_size_numbers` |
| Entscheidet n oder die Zahl bestimmter Richtungen? | J = 50: n = 24 / 48 / 96: sd-Verhältnis **0,98 / 0,68 / 0,64**. Die Daten bestimmen bei Rauschen 1 % 24 / 30 / 31 Richtungen (Richtungen mit prior-gewichtetem Hesse-Eigenwert über 1, Stück 8); bei n = 48 sind es 21 / 30 / 38 bei Rauschen 10 / 1 / 0,1 %. | `test_dimension_numbers` |
| Was rettet ein kleines Ensemble (Experiment 2, Gaspari-Cohn auf C_xg und C_gg, Rauschen 1 %)? | J = 10: ohne Lokalisierung 0,02 und 3,0 sd; Halbbreite 0,4 / **0,2** / 0,1 / 0,05: sd-Verhältnis 0,38 / **0,87** / 1,31 / 1,45, Mittelwert 1,2 / **0,94** / 0,97 / 1,03 sd. J = 25 (Halbbreite 0,4): 0,76 und 0,62 sd; J = 50: 0,90; J = 100: 0,98. Eine zu kleine Halbbreite macht die Unsicherheit zu groß (1,5 bis 1,6). | `test_localization_numbers` |
| Wie viele Schritte braucht die nichtlineare Welt (Experiment 3, J = 400, gegen MCMC)? | sd-Verhältnis bei 1 / 4 / 16 Schritten: ρ = 0: 0,96 / 0,93 / 0,90; **ρ = 1: 3,17 / 0,90 / 0,83**; ρ = 2: 4,2 / 1,14 / 0,76; ρ = 4: 4,4 / 2,5 / 0,62. Das einzelne Update ist bei ρ = 1 zu breit (Mittel 2,5 sd daneben, Fehler 67,5 % gegen 13,5 % beim MCMC-Mittel); 4 Schritte treffen (Mittel 0,37 sd, Fehler 13,3 %). Die Laplace-Näherung liegt gegen MCMC bei 0,99 bis 1,00. | `test_steps_numbers` |
| Hilft Iterieren im linearen Fall (J = 100 / 400, α = N)? | Nein: das sd-Verhältnis **sinkt** mit der Schrittzahl (J = 100: 0,85 / 0,80 / 0,74 / 0,69 / 0,65 bei N = 1 / 2 / 4 / 8 / 16; J = 400: 0,96 … 0,90), das Mittel bleibt (0,45 bis 0,48 sd). | `test_linear_tempering_numbers` |
| Als ableitungsfreier Löser (Experiment 4, α = 1, Rauschen 0,1 %, bis 30 Iterationen)? | **ρ = 1:** J = 20 und 50 bleiben stehen (Fehler 63 und 67 %, mittlere Streuung nach wenigen Iterationen unter 0,01, also unter 2 % des Priors); **J = 200:** Diskrepanzprinzip im Median nach 4,5 Iterationen, Fehler **15,5 %** (MAP 9,2 % mit rund 9 Jacobi-Matrizen), nach 30 Iterationen 21,9 %. **ρ = 2:** J = 50 läuft in mehreren der 10 Wiederholungen davon (in dieser Messung 6; Fehler weit über dem Zehnfachen der Signalnorm, die Zahl der Fälle ist plattformabhängig), J = 200 hält nach 7 Iterationen bei 20,8 %. | `test_solver_numbers` |
| Einzelstichproben (Presets) | **Standard** (ρ = 1, Tempering, 4 Schritte, J = 400): Mittel 0,40 sd, sd-Verhältnis 0,92, Fehler 15,0 % gegen 17,8 % beim Referenzmittel. **Kleines Ensemble** (ρ = 0, J = 25): 0,23 und 5,5 sd. **Lokalisiert** (J = 10, Halbbreite 0,2): 0,83 und 1,0 sd (ohne: 0,02 und 3,3 sd). **Update bei ρ = 1:** 3,28 und 2,5 sd. **Löser-Modus:** Halt nach 5 Iterationen, 15,6 % gegen 10,0 % beim MAP. | `test_preset_numbers` |
| Mini-Instanz von Hand | 3 × 3-Matrix, handgebautes Ensemble ±e₁, ±e₂, ±e₃ (C_xx = 0,4 I): Verstärkung mit den Eigenwerten 0,97561 / 1,81818 / 2,857143, Mittel nach dem Update (1,205575, 1,491289, 1,205575); für α = 4 (1,181818, 1,272727, 1,181818); mit dem Taper I (nur Diagonalen) (1,608247, 1,534247, 1,608247); exakter Posterior (1,153465, 1,653465, 1,153465). | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Gliederung stellte „Unsicherheit nur im Gaußschen Fall“ als eine Einschränkung dar; die Messung zeigt zwei.** Auch im linearen Fall ist das Update nur im Grenzwert J → ∞ der exakte Posterior (Fehler ∝ 1/√J). Im nichtlinearen Fall ist die Streuung des einzelnen Updates **zu groß** (3 bis 4-fach) und die vieler Schritte **zu klein**.
- **Nicht n zählt, sondern die Zahl der von den Daten bestimmten Richtungen.** J = 50 reicht bei n = 24 (alle 24 Richtungen bestimmt), nicht bei n = 48 und 96 (30 und 31).
- **Lokalisierung muss beide Kovarianzen dämpfen.** In der Vormessung wurde nur C_xg gedämpft: das sd-Verhältnis stieg auf 4 bis 500, weil der Kreuzterm nicht mehr zu C_gg passt.
- **Iterieren gehört nicht zum Posterior-Verfahren im linearen Fall.** Mit exakten Momenten ist α = 1 wiederholt Tikhonov mit λ = σ/(τ√k) (Stück 2), nicht der Posterior; das Tempering mit α = N ist die Posterior-Fassung.
- **Ein zusammengebrochenes Ensemble sieht fertig aus.** Bei J = 20 steht das Mittel nach wenigen Iterationen (Fehler 63 %), die mittlere Streuung ist auf unter 0,01 (2 % des Priors) gefallen, und kein Warnzeichen erscheint außer dem Streuungsmaß selbst.
- **Ein Verdikt für das Update gilt nicht für den Löser.** Im Löser-Modus gibt es keinen Posterior; das Verdikt dort vergleicht den Fehler mit dem des MAP und prüft den Halt.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ensemble ist groß genug** | Bei Rauschen 1 % und J = 25 liegt das sd-Verhältnis bei 0,20, bei J = 10 bei 0,02; das Mittel liegt 2,7 sd daneben. | Lokalisierung (Experiment 2) |
| **Die Abbildung ist nahezu linear** | Bei ρ = 1 ist das einzelne Update zu breit (sd-Verhältnis 3,17 gegen MCMC). | Stück 9 (Gauss-Newton) |
| **Der Posterior ist gaußsch** | Auch viele getemperte Schritte unterschätzen die Streuung (ρ = 4, 16 Schritte: 0,62); die Näherung kennt keine Schiefe. | Stück 15 (Diffusions-Prior) |
| **Die Referenz stimmt** | Bei ρ > 0 ist die Referenz in der Instanzansicht die Laplace-Näherung; sie stimmt mit MCMC bei ρ ≤ 2 auf 1 % überein, bei ρ = 4 weniger. | – |
| **Ein Modell, ein Prior, ein Gitter** | Wärmeleitung und logistische Reaktion (Stück 9), exponentieller Prior, eine Dimension; die Lokalisierung nutzt, dass der Operator lokal ist. | – |
| **Störbeobachtungen** | Ein Wurzel-Filter (ETKF) würde die Stichproben-Störung der Beobachtungen sparen; nicht gebaut. | – |
| **Der Löser braucht kein Streuungsmaß** | Im Löser-Modus gibt es keinen Posterior; ein zusammengebrochenes Ensemble sieht fertig aus. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [nichtlinear-inverse-demo](https://github.com/sebastian-hanisch/nichtlinear-inverse-demo): Stück 9; dasselbe Modell, die Jacobi-Matrix und die Laplace-Näherung sind hier Referenz und Vergleich.
- [posterior-sampling-demo](https://github.com/sebastian-hanisch/posterior-sampling-demo): Stück 8; das vorkonditionierte Zufallsschritt-Metropolis-Verfahren und die Identifikation (bestimmte Richtungen) kehren im Experiment 1 und 3 wieder.
- [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo): Stück 7; der exakte Gauß-Posterior im linearen Fall.
- [kalman-filter-demo](https://github.com/sebastian-hanisch/kalman-filter-demo): das Kalman-Update dort sequentiell über die Zeit, hier ein Block von Parametern mit Kovarianzen aus einem Ensemble; [particle-filter-demo](https://github.com/sebastian-hanisch/particle-filter-demo) nennt Ensemble-Filter als nicht umgesetzt.

## Tests

Mini-Instanz von Hand (3 × 3: Wärmeleitung, Modell, Jacobi-Matrix, Gaspari-Cohn-Funktion und Taper, Kalman-Schritt am handgebauten Ensemble für α = 1 und α = 4 und mit Taper, Störung mit √α, Start-Ensemble und Halt in `run_eki`, Statistik, exakter Posterior, MAP, Laplace-Näherung, Negativ-Log-Posterior, Sampler und R̂), Orakel (exakte Kalman-Verstärkung gegen die Informationsform, Tempering mit exakten Momenten gleich Posterior für N = 1, 2, 4, 16, Iteration mit exakten Momenten gleich Tikhonov mit λ = σ/(τ√k), Ensemble gegen den exakten Posterior mit Konvergenz ∝ 1/√J, Teilraum-Eigenschaft, Stichproben-Kovarianzen gegen `np.cov`, Lokalisierung im Grenzfall und gegen ein kleines Ensemble, MCMC gegen den exakten Posterior und die Laplace-Näherung), Generator, Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler-Grenzen, jedes Verfahren und jede Datenart, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, Gitter) |
| `eki_model.py` | Modell aus Stück 9, Ensemble-Kalman-Schritt, Lauf, Lokalisierung, exakter Posterior, MAP, Laplace-Näherung, Sampler |
| `eki_evaluation.py` | Instanz, Analyse, Experimente, Verdikte |
| `eki_visualization.py`, `eki_pdf_export.py` | Figuren, PDF |
| `eki_constants.py`, `eki_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Wurzel-Filter (ETKF, EAKF) und adaptive Inflation:** nur Störbeobachtungen; das Wurzel-Filter spart die Stichproben-Störung, nicht die Ensemblegröße.
- **Varianz-Regularisierung und Konvergenzbeweise für den Löser-Modus** (Hanke-artig): der Halt nach dem Diskrepanzprinzip genügt für die Aussage.
- **Mehrdimensionale Lokalisierung, adaptive Halbbreite:** eine Dimension, feste Halbbreite je Lauf.
- **Mehrgipflige Posteriors:** dort fällt die Gaußsche Näherung aus (Stück 15).
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
