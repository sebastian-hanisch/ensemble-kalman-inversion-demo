"""Aussagen aus README, Verdikten und Hilfetexten, einzeln an den festen Strömen gemessen (Seeds ab SEED_BASE). Ändert sich eine Zahl, muss der Text mit.
Linux-CI: in der letzten Stelle nichts festlegen; chaotische Zellen (Ensemble läuft davon) nur mit Schranken."""

import pytest

import eki_constants as C
import eki_evaluation as ev


@pytest.fixture(scope="module")
def size():
    return ev.size_experiment()


@pytest.fixture(scope="module")
def loc():
    return ev.localization_experiment()


@pytest.fixture(scope="module")
def steps():
    return ev.steps_experiment()


@pytest.fixture(scope="module")
def solver():
    return ev.solver_experiment()


# ----------------------------------------------------------------------------- Presets

def test_preset_numbers():
    std = ev.analyse(ev.Settings(**C.PRESETS["Standard"]))
    assert std.mean_err_sd == pytest.approx(0.40, abs=0.1) and std.sd_ratio == pytest.approx(0.92, abs=0.04) and std.err == pytest.approx(0.150, abs=0.01) and std.err_ref == pytest.approx(0.178, abs=0.005)
    lin = ev.analyse(ev.Settings(**C.PRESETS["Lineares Update (ρ = 0)"]))
    assert lin.sd_ratio == pytest.approx(0.98, abs=0.03) and lin.mean_err_sd == pytest.approx(0.17, abs=0.06) and lin.ref_kind == "exakt"
    small = ev.analyse(ev.Settings(**C.PRESETS["Kleines Ensemble"]))
    assert small.sd_ratio == pytest.approx(0.23, abs=0.06) and small.mean_err_sd == pytest.approx(5.5, abs=1.5) and small.err > 1.0
    locd = ev.analyse(ev.Settings(**C.PRESETS["Kleines Ensemble, lokalisiert"]))
    assert locd.sd_ratio == pytest.approx(0.83, abs=0.1) and locd.mean_err_sd == pytest.approx(1.03, abs=0.35)
    nolocd = ev.analyse(ev.Settings(**dict(C.PRESETS["Kleines Ensemble, lokalisiert"], loc=0.0)))
    assert nolocd.sd_ratio == pytest.approx(0.02, abs=0.02) and nolocd.mean_err_sd == pytest.approx(3.3, abs=0.8)
    upd = ev.analyse(ev.Settings(**C.PRESETS["Update bei Nichtlinearität"]))
    assert upd.sd_ratio == pytest.approx(3.28, abs=0.4) and upd.mean_err_sd == pytest.approx(2.54, abs=0.5) and upd.err == pytest.approx(0.716, abs=0.08)
    sol = ev.analyse(ev.Settings(**C.PRESETS["Löser-Modus"]))
    assert 3 <= sol.stop <= 8 and sol.err == pytest.approx(0.156, abs=0.04) and sol.err_map == pytest.approx(0.100, abs=0.01) and sol.map_jacs >= 5
    run = ev.analyse(ev.Settings(**C.PRESETS["Ensemble läuft davon"]))
    assert run.diverged and run.err > 100.0 and run.sd.mean() < 1e-3


# ----------------------------------------------------------------------------- Experiment 1

def test_size_numbers(size):
    rows = {(r["noise"], r["j"]): r for r in size["rows"]}
    ratios = {nz: [rows[(nz, j)]["ratio"] for j in C.SIZE_J] for nz in C.SIZE_NOISE}
    assert ratios[1e-2] == pytest.approx([0.02, 0.20, 0.67, 0.86, 0.96, 0.99, 1.00], abs=0.04)
    assert ratios[1e-1] == pytest.approx([0.16, 0.55, 0.79, 0.91, 0.97, 0.99, 1.00], abs=0.05) and ratios[1e-3] == pytest.approx([0.00, 0.03, 0.56, 0.81, 0.95, 0.99, 1.00], abs=0.05)
    dm = [rows[(1e-2, j)]["dm"] for j in C.SIZE_J]
    assert dm == pytest.approx([3.36, 2.65, 0.79, 0.46, 0.21, 0.09, 0.05], rel=0.25, abs=0.03) and 1.8 < dm[4] / dm[5] < 2.8 and 1.5 < dm[5] / dm[6] < 2.8
    assert [rows[(nz, 10)]["directions"] for nz in C.SIZE_NOISE] == [21, 30, 38] and rows[(1e-2, 10)]["err_exact"] == pytest.approx(0.120, abs=0.003)
    assert rows[(1e-2, 6400)]["err"] == pytest.approx(0.120, abs=0.004) and rows[(1e-2, 10)]["err"] > 0.5
    assert all(rows[(nz, 50)]["ratio"] < rows[(nz, 400)]["ratio"] < rows[(nz, 6400)]["ratio"] + 0.01 for nz in C.SIZE_NOISE)
    text = ev.verdict_size(size)[1]
    assert "0,02 / 0,20 / 0,67 / 0,86 / 0,96" in text and "Von 48 Richtungen bestimmen die Daten 30; bei Rauschen 0,1 % sind es 38" in text and "J = 50 trifft die Streuung nur zu 0,56" in text


def test_dimension_numbers(size):
    dims = {(d["n"], d["j"]): d for d in size["dims"]}
    assert dims[(24, 50)]["ratio"] == pytest.approx(0.98, abs=0.03) and dims[(24, 50)]["dm"] == pytest.approx(0.19, abs=0.06) and dims[(24, 25)]["dm"] > 1.5
    assert dims[(48, 50)]["ratio"] == pytest.approx(0.68, abs=0.05) and dims[(96, 50)]["ratio"] == pytest.approx(0.64, abs=0.05) and dims[(48, 50)]["dm"] > 0.6 and dims[(96, 50)]["dm"] > 0.6
    assert dims[(48, 192)]["ratio"] == pytest.approx(0.92, abs=0.04) and dims[(96, 384)]["ratio"] == pytest.approx(0.96, abs=0.03)
    assert [dims[(n, 50)]["directions"] for n in C.DIM_NS] == [24, 30, 31]                  # maßgeblich ist die Zahl bestimmter Richtungen, nicht n


# ----------------------------------------------------------------------------- Experiment 2

def test_localization_numbers(loc):
    rows = {(r["j"], r["c"]): r for r in loc["rows"]}
    assert (rows[(10, 0.0)]["ratio"], rows[(10, 0.0)]["dm"]) == pytest.approx((0.02, 3.03), abs=0.5) and (rows[(10, 0.2)]["ratio"], rows[(10, 0.2)]["dm"]) == pytest.approx((0.87, 0.94), abs=0.12)
    assert rows[(10, 0.4)]["ratio"] == pytest.approx(0.38, abs=0.06) and rows[(10, 0.1)]["ratio"] == pytest.approx(1.31, abs=0.08) and rows[(10, 0.05)]["ratio"] == pytest.approx(1.45, abs=0.08)
    assert rows[(25, 0.4)]["ratio"] == pytest.approx(0.76, abs=0.06) and rows[(25, 0.4)]["dm"] == pytest.approx(0.62, abs=0.1) and rows[(50, 0.4)]["ratio"] == pytest.approx(0.90, abs=0.05)
    assert rows[(100, 0.4)]["ratio"] == pytest.approx(0.98, abs=0.04) and rows[(100, 0.05)]["ratio"] == pytest.approx(1.59, abs=0.08)
    assert all(rows[(j, 0.05)]["ratio"] > 1.25 and rows[(j, 0.0)]["ratio"] < 1.0 for j in C.LOC_J)                 # zu kleine Halbbreite: zu breit
    assert rows[(10, 0.2)]["dm"] < rows[(10, 0.0)]["dm"] and rows[(25, 0.4)]["dm"] < rows[(25, 0.0)]["dm"] and rows[(50, 0.4)]["dm"] < rows[(50, 0.0)]["dm"]
    text = ev.verdict_localization(loc)[1]
    assert "ohne Lokalisierung das sd-Verhältnis 0,0" in text and "mit der Halbbreite 0,2 sind es 0,8" in text and "bei 0,05 ist das Verhältnis 1,4" in text


# ----------------------------------------------------------------------------- Experiment 3

def test_steps_numbers(steps):
    rows = {r["rho"]: r for r in steps["rows"]}
    assert [round(rows[r]["laplace_ratio"], 2) for r in C.STEPS_RHOS] == pytest.approx([0.99, 1.0, 1.0, 1.0], abs=0.02) and all(rows[r]["rhat"] < 1.1 for r in C.STEPS_RHOS)
    assert [rows[r]["by_n"][1]["ratio"] for r in C.STEPS_RHOS] == pytest.approx([0.96, 3.17, 4.2, 4.41], abs=0.3)
    assert [rows[r]["by_n"][4]["ratio"] for r in C.STEPS_RHOS] == pytest.approx([0.93, 0.90, 1.14, 2.53], abs=0.1)
    assert [rows[r]["by_n"][16]["ratio"] for r in C.STEPS_RHOS] == pytest.approx([0.90, 0.83, 0.76, 0.62], abs=0.04)
    assert rows[1.0]["by_n"][1]["dm"] == pytest.approx(2.45, abs=0.3) and rows[1.0]["by_n"][4]["dm"] == pytest.approx(0.37, abs=0.1) and rows[1.0]["by_n"][16]["dm"] == pytest.approx(0.34, abs=0.1)
    assert rows[2.0]["by_n"][4]["dm"] == pytest.approx(0.73, abs=0.15) and rows[4.0]["by_n"][16]["dm"] == pytest.approx(0.33, abs=0.1)
    assert rows[1.0]["by_n"][1]["err"] == pytest.approx(0.675, abs=0.08) and rows[1.0]["by_n"][4]["err"] == pytest.approx(0.133, abs=0.015) and rows[1.0]["err_mcmc"] == pytest.approx(0.135, abs=0.005)
    assert rows[4.0]["err_mcmc"] == pytest.approx(0.579, abs=0.03) and rows[4.0]["err_map"] == pytest.approx(0.623, abs=0.03)
    assert all(0.2 < rows[r]["acc"] < 0.3 for r in C.STEPS_RHOS) and all(rows[r]["by_n"][16]["ratio"] < rows[r]["by_n"][4]["ratio"] for r in C.STEPS_RHOS)
    text = ev.verdict_steps(steps)[1]
    assert "gegen MCMC" in text and "vier Schritte treffen (0,9" in text and "bei ρ = 4 sind es 0,6" in text and "zu breit (sd-Verhältnis 3,1" in text


def test_linear_tempering_numbers(steps):
    lin = {(d["j"], d["n"]): d for d in steps["linear"]}
    assert [lin[(100, n)]["ratio"] for n in C.TEMPER_N] == pytest.approx([0.85, 0.80, 0.74, 0.69, 0.65], abs=0.04)
    assert [lin[(400, n)]["ratio"] for n in C.TEMPER_N] == pytest.approx([0.96, 0.95, 0.94, 0.92, 0.90], abs=0.03)
    assert all(lin[(j, 16)]["ratio"] < lin[(j, 1)]["ratio"] for j in C.TEMPER_J) and all(abs(lin[(j, 16)]["dm"] - lin[(j, 1)]["dm"]) < 0.1 for j in C.TEMPER_J)


# ----------------------------------------------------------------------------- Experiment 4

def test_solver_numbers(solver):
    rows = {(r["rho"], r["j"]): r for r in solver["rows"]}
    small, mid, big = rows[(1.0, 20)], rows[(1.0, 50)], rows[(1.0, 200)]
    assert small["stop_found"] == 0 and mid["stop_found"] == 0 and small["err_final"] == pytest.approx(0.63, abs=0.08) and mid["err_final"] == pytest.approx(0.67, abs=0.1)
    assert small["spreads"][0] > 0.4 and small["spreads"][3] < 0.01 and mid["spreads"][5] < 0.01 and big["spreads"][5] > 0.01               # kleine Ensembles brechen zusammen
    assert big["stop_found"] == solver["reps"] and 3.0 <= big["stop"] <= 6.0 and big["err_stop"] == pytest.approx(0.155, abs=0.03) and big["err_final"] == pytest.approx(0.219, abs=0.04)
    assert big["err_stop"] > big["err_map"] > 0.05 and big["err_final"] > big["err_stop"] and big["err_map"] == pytest.approx(0.092, abs=0.005) and big["map_jacs"] >= 5
    assert big["evals"] == int(big["stop"] * 200) and big["resid"][5] < 1.5 < big["resid"][1]
    two_small, two_mid, two_big = rows[(2.0, 20)], rows[(2.0, 50)], rows[(2.0, 200)]
    assert two_small["stop_found"] == 0 and two_small["err_final"] == pytest.approx(0.90, abs=0.1) and two_mid["stop_found"] == 0 and two_mid["diverged"] >= 3 and two_mid["err_final"] > 3.0
    assert 5.0 <= two_big["stop"] <= 9.0 and two_big["err_stop"] == pytest.approx(0.208, abs=0.04) and two_big["diverged"] == 0
    text = ev.verdict_solver(solver)[1]
    assert "mit 200 Mitgliedern im Median nach " in text and " Iterationen bei einem Fehler von 15" in text and "steht das Ensemble bei 6" in text and "von 10 Wiederholungen davon" in text
