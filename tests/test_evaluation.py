"""Auswertung: Formatierung, Instanz, Analyse, Verdikte (Art und Text, Schwellen an handgesetzten Werten), Form der Experimente und ihre Reproduzierbarkeit."""

import dataclasses
import math

import numpy as np
import pytest

import eki_constants as C
import eki_evaluation as ev
import eki_model as model
from inverse_generator import SplitMix64


# ----------------------------------------------------------------------------- Formatierung

def test_formatting_helpers():
    assert ev.de(-1.234, 2) == "−1,23" and ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(3.0e-12) == "3,0·10⁻¹²" and ev.share(0.1525) == "15 %"
    assert ev.sci(0.0) == "0,0" and ev.sci(float("inf")) == "unendlich" and ev.sci(9.96) == "1,0·10¹"
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0,01 %", "0,1 %", "1 %", "3 %", "10 %"]
    assert [ev.rho_label(v) for v in C.RHO_CHOICES] == ["0", "0,5", "1", "2", "4"] and ev.rho_label(10.0) == "10"
    assert [ev.loc_label(v) for v in C.LOC_CHOICES] == ["aus", "0,05", "0,10", "0,20", "0,40"]
    assert ev.count_text(1234567) == "1.234.567" and ev.count_text(5) == "5" and ev.count_text(1000) == "1.000"
    assert ev.num_text(1000.0) == "1,0·10³" and ev.num_text(999.99) == "1000,0" and ev.num_text(0.01) == "0,01" and ev.num_text(0.0099) == "9,9·10⁻³" and ev.num_text(10.0) == "10,0"
    assert ev.num_text(9.99) == "9,99" and ev.num_text(0.0) == "0,00" and ev.num_text(float("inf")) == "unendlich"


def test_err_pct_digits_and_boundaries():
    assert [ev.err_pct(v) for v in (2.58, 0.1188, 0.0399, 0.0087, 8.7e-4, 4471.0, 0.57)] == ["258 %", "11,9 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach", "57,0 %"]
    assert ev.err_pct(10.0) == "1,0·10¹-fach" and ev.err_pct(9.99) == "999 %" and ev.err_pct(0.995) == "100 %" and ev.err_pct(0.994) == "99,4 %"
    assert ev.err_pct(0.0095) == "0,9 %" and ev.err_pct(0.0094) == "0,94 %" and ev.err_pct(0.0001) == "0,01 %" and ev.err_pct(0.00009) == "9,0·10⁻³ %" and ev.err_pct(float("inf")) == "unendlich"


def test_rel_err_by_hand():
    assert ev.rel_err(np.array([3.0, 4.0]), np.array([0.0, 5.0])) == pytest.approx(math.sqrt(10.0) / 5.0) and ev.rel_err(np.ones(3), np.ones(3)) == 0.0


# ----------------------------------------------------------------------------- Instanz

def test_instance_is_noise_on_the_fine_or_the_inverted_model():
    s, x, b, sigma = ev.instance(24, 0.03, 1.0, 1e-2, "honest", 5)
    fine = model.Model(model.heat_matrix(24, 0.03, C.K_TRUE), C.K_TRUE, 1.0).forward(x)
    assert s.shape == x.shape == b.shape == (24,) and sigma == pytest.approx(1e-2 * float(np.sqrt(np.mean(fine ** 2)))) and b - fine == pytest.approx(SplitMix64(5).normal(24) * sigma)
    s, x, b2, sigma2 = ev.instance(24, 0.03, 1.0, 1e-2, "crime", 5)
    crime = ev.inverse_model(24, 0.03, 1.0).forward(x)
    assert b2 - crime == pytest.approx(SplitMix64(5).normal(24) * sigma2) and not np.allclose(b, b2, atol=1e-4)
    assert np.array_equal(ev.instance(24, 0.03, 1.0, 1e-2, "honest", 5)[2], b) and not np.array_equal(ev.instance(24, 0.03, 1.0, 1e-2, "honest", 6)[2], b)


def test_problem_and_reference():
    inv, x, b, sigma, pcov, lc = ev.problem(1.0, 1e-2)
    assert b == pytest.approx(ev.instance(48, 0.03, 1.0, 1e-2, "honest", C.SEED_BASE + 1)[2]) and lc @ lc.T == pytest.approx(pcov) and inv.k == C.K_INV and inv.rho == 1.0
    mean, cov, kind, x_map, iters, jacs = ev.reference(inv, b, sigma, pcov, lc)
    assert kind == "Laplace-Näherung" and mean is x_map and cov == pytest.approx(model.laplace_covariance(inv, x_map, sigma, pcov)) and jacs >= iters >= 1
    lin = ev.problem(0.0, 1e-2)
    mean0, cov0, kind0, x_map0, _, _ = ev.reference(lin[0], lin[2], lin[3], lin[4], lin[5])
    assert kind0 == "exakt" and mean0 == pytest.approx(model.linear_posterior(lin[0].total_matrix(), lin[4], lin[3], lin[2])[0]) and x_map0 == pytest.approx(mean0, abs=1e-6)


def test_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


# ----------------------------------------------------------------------------- Analyse

def test_analysis_fields_of_the_standard_instance():
    a = ev.analyse(ev.Settings())
    assert a.steps_run == 4 and a.alpha == 4.0 and a.evals == 1600 and a.finite and not a.diverged and a.stop is None and a.ref_kind == "Laplace-Näherung"
    assert a.mean.shape == a.sd.shape == a.ref_mean.shape == a.ref_sd.shape == (48,) and a.members.shape == (C.N_MEMBERS_SHOWN, 48)
    assert len(a.errs) == len(a.spreads) == len(a.resid) == 5 and a.errs[0] == pytest.approx(ev.rel_err(np.zeros(48), a.x), abs=0.1) and a.err == pytest.approx(a.errs[-1])
    assert a.err_ref == pytest.approx(ev.rel_err(a.ref_mean, a.x)) and a.err_map == pytest.approx(ev.rel_err(a.x_map, a.x)) and a.prior_spread == pytest.approx(C.PRIOR_TAU, abs=1e-4)
    assert a.sd_ratio_min <= a.sd_ratio <= a.sd_ratio_max and a.mean_err_sd >= 0.0 and a.map_jacs >= a.map_iters >= 1


def test_method_determines_steps_alpha_and_stop():
    upd = ev.analyse(ev.Settings(method="update", steps=16))
    assert upd.steps_run == 1 and upd.alpha == 1.0 and upd.evals == 400 and upd.stop is None
    tem = ev.analyse(ev.Settings(method="tempering", steps=8))
    assert tem.steps_run == 8 and tem.alpha == 8.0 and tem.evals == 3200
    it = ev.analyse(ev.Settings(method="iteration", steps=30, j=200, noise=1e-3))
    assert it.steps_run == 30 and it.alpha == 1.0 and it.stop is not None and 1 <= it.stop <= 30 and it.mean == pytest.approx(it.members.mean(axis=0), abs=1.0)
    full = model.run_eki(ev.inverse_model(48, 0.03, 1.0), ev.instance(48, 0.03, 1.0, 1e-3, "honest", 11)[2], ev.instance(48, 0.03, 1.0, 1e-3, "honest", 11)[3],
                         np.linalg.cholesky(model.prior_cov(ev.instance(48, 0.03, 1.0, 1e-3, "honest", 11)[0])), 200, 30, 1.0, 18, None, C.DISC_FACTOR)
    assert it.stop == full.stop and it.mean == pytest.approx(full.ens_stop.mean(axis=0)) and len(it.errs) == 31


def test_linear_case_uses_the_exact_reference_and_localization_is_passed_on():
    a = ev.analyse(ev.Settings(rho=0.0, method="update", steps=1))
    assert a.ref_kind == "exakt" and a.mean_err_sd < 0.5 and 0.9 < a.sd_ratio < 1.05
    plain = ev.analyse(ev.Settings(rho=0.0, method="update", j=10))
    loc = ev.analyse(ev.Settings(rho=0.0, method="update", j=10, loc=0.2))
    assert loc.sd_ratio > 10.0 * plain.sd_ratio and loc.mean_err_sd < plain.mean_err_sd


def test_diverged_analysis_keeps_the_last_finite_ensemble():
    a = ev.analyse(ev.Settings(**C.PRESETS["Ensemble läuft davon"]))
    assert a.diverged and a.err > C.DIVERGE_ERR and a.finite and np.all(np.isfinite(a.mean)) and a.stop is None


# ----------------------------------------------------------------------------- Verdikte

def _variant(a, **changes):
    return dataclasses.replace(a, **changes)


def test_verdict_instance_success_and_warning_branches_for_the_posterior_methods():
    a = ev.analyse(ev.Settings())
    kind, text = ev.verdict_instance(a)
    assert kind == "success" and "Mittel und Streuung stimmen mit der Referenz überein" in text and "Tempering, 400 Mitglieder, 4 Schritte (1.600 Vorwärtsläufe)" in text
    assert ev.verdict_instance(_variant(a, mean_err_sd=C.MEAN_OK))[0] == "success" and ev.verdict_instance(_variant(a, mean_err_sd=C.MEAN_OK + 0.01))[0] == "warning"
    assert ev.verdict_instance(_variant(a, sd_ratio=C.SD_LO))[0] == "success" and ev.verdict_instance(_variant(a, sd_ratio=C.SD_LO - 0.01))[0] == "warning"
    assert ev.verdict_instance(_variant(a, sd_ratio=C.SD_HI))[0] == "success" and ev.verdict_instance(_variant(a, sd_ratio=C.SD_HI + 0.01))[0] == "warning"
    assert "Streuung ist zu klein: sd-Verhältnis 0,50 (unter 0,80)" in ev.verdict_instance(_variant(a, sd_ratio=0.5))[1]
    assert "Streuung ist zu groß: sd-Verhältnis 3,00 (über 1,25)" in ev.verdict_instance(_variant(a, sd_ratio=3.0))[1]
    assert "weicht um 2,00 sd von der Referenz ab" in ev.verdict_instance(_variant(a, mean_err_sd=2.0))[1]
    kind, text = ev.verdict_instance(_variant(a, diverged=True, err=5.0))
    assert kind == "warning" and "davongelaufen" in text and "Streuung ist" not in text


def test_verdict_instance_for_the_iteration_method():
    a = ev.analyse(ev.Settings(method="iteration", steps=30, j=200, noise=1e-3))
    kind, text = ev.verdict_instance(a)
    assert kind == "success" and "Löser hält nach dem Diskrepanzprinzip" in text and f"Halt nach {a.stop} Iterationen" in text and "Jacobi-Matrizen" in text
    assert ev.verdict_instance(_variant(a, err=C.SOLVER_FACTOR * a.err_map))[0] == "success" and ev.verdict_instance(_variant(a, err=C.SOLVER_FACTOR * a.err_map * 1.01))[0] == "warning"
    kind, text = ev.verdict_instance(_variant(a, stop=None, resid=[1.0, 7.0], steps_run=30))
    assert kind == "warning" and "nach 30 Iterationen nicht erreicht" in text and "das 7,00-Fache" in text and "kein Halt" in text
    assert "Fehler" in ev.verdict_instance(_variant(a, err=3.0 * a.err_map))[1] and "gegen" in ev.verdict_instance(_variant(a, err=3.0 * a.err_map))[1]


# ----------------------------------------------------------------------------- Experimente (Form und Reproduzierbarkeit)

def test_size_experiment_shape_and_reproducibility():
    res = ev.size_experiment(reps=2)
    assert len(res["rows"]) == len(C.SIZE_NOISE) * len(C.SIZE_J) and [(r["noise"], r["j"]) for r in res["rows"]][:2] == [(0.1, 10), (0.1, 25)] and len(res["dims"]) == 9
    for r in res["rows"]:
        assert r["dm_min"] <= r["dm"] <= r["dm_max"] and r["ratio_min"] <= r["ratio"] <= r["ratio_max"] and 0.0 < r["err_exact"] < 0.5 and 10 <= r["directions"] <= 48
    assert [d["j"] for d in res["dims"][:3]] == [50, 25, 96] and res == ev.size_experiment(reps=2)
    assert ev.verdict_size(res)[0] == "warning"


def test_localization_experiment_shape_and_reproducibility():
    res = ev.localization_experiment(reps=2)
    assert len(res["rows"]) == len(C.LOC_J) * len(C.LOC_RADII) and res["rows"][0]["j"] == 10 and res["rows"][0]["c"] == 0.0 and res == ev.localization_experiment(reps=2)
    assert ev.verdict_localization(res)[0] == "info"


def test_steps_experiment_shape_and_reproducibility(monkeypatch):
    monkeypatch.setattr(C, "STEPS_REPS", 2)
    monkeypatch.setattr(C, "TEMPER_REPS", 2)
    res = ev.steps_experiment(chains=4, mcmc_steps=1500)
    assert [r["rho"] for r in res["rows"]] == list(C.STEPS_RHOS) and all(set(r["by_n"]) == set(C.STEPS_N) for r in res["rows"]) and len(res["linear"]) == 10
    assert res["chains"] == 4 and res["steps"] == 1500 and all(0.1 < r["acc"] < 0.5 for r in res["rows"]) and res == ev.steps_experiment(chains=4, mcmc_steps=1500)
    assert ev.verdict_steps(res)[0] == "warning"


def test_solver_experiment_shape():
    res = ev.solver_experiment()
    assert len(res["rows"]) == len(C.SOLVER_RHOS) * len(C.SOLVER_J) and all(len(r["errs"]) == C.SOLVER_ITERS + 1 for r in res["rows"])
    r = res["rows"][-1]
    assert r["j"] == 200 and r["rho"] == 2.0 and r["stop_found"] == res["reps"] and r["stop"] is not None and r["evals"] == int(r["stop"] * r["j"])
    assert res["rows"][0]["stop_found"] == 0 and res["rows"][0]["stop"] is None and res["rows"][0]["evals"] == C.SOLVER_ITERS * 20
    assert ev.verdict_solver(res)[0] == "warning"


def test_verdicts_of_the_experiments_on_hand_built_results():
    rows = [{"noise": nz, "j": j, "ratio": 0.1 * k, "dm": 1.0 / (k + 1), "directions": 30 if nz == 1e-2 else 38} for nz in C.SIZE_NOISE for k, j in enumerate(C.SIZE_J, start=1)]
    kind, text = ev.verdict_size({"rows": rows, "n": 48})
    assert kind == "warning" and "J = 10 / 25 / 50 / 100 / 400 bei 0,10 / 0,20 / 0,30 / 0,40 / 0,50" in text and "Von 48 Richtungen bestimmen die Daten 30" in text and "bei Rauschen 0,1 % sind es 38" in text
    steps = {"rows": [{"rho": rho, "by_n": {n: {"ratio": 3.0 / n, "dm": 1.0 / n} for n in C.STEPS_N}} for rho in C.STEPS_RHOS]}
    kind, text = ev.verdict_steps(steps)
    assert kind == "warning" and "sd-Verhältnis 3,00 gegen MCMC, Mittel 1,00 sd daneben" in text and "(0,75, 0,25 sd)" in text and "sechzehn unterschätzen die Streuung (0,19)" in text
    loc = {"rows": [{"j": 10, "c": c, "ratio": 0.5 + c, "dm": 2.0 - c} for c in C.LOC_RADII]}
    assert "ohne Lokalisierung das sd-Verhältnis 0,50 und liegt 2,00 sd" in ev.verdict_localization(loc)[1] and "0,70 und 1,80 sd" in ev.verdict_localization(loc)[1]


def test_experiment_rows_keep_min_below_median_below_max(monkeypatch):
    for r in ev.localization_experiment(reps=3)["rows"]:
        assert r["ratio_min"] <= r["ratio"] <= r["ratio_max"]
    monkeypatch.setattr(C, "STEPS_REPS", 3)
    monkeypatch.setattr(C, "TEMPER_REPS", 2)
    res = ev.steps_experiment(chains=4, mcmc_steps=1000)
    for row in res["rows"]:
        for v in row["by_n"].values():
            assert v["ratio_min"] <= v["ratio"] <= v["ratio_max"]


def test_solver_experiment_pads_runs_that_stop_early_with_nan(monkeypatch):
    monkeypatch.setattr(C, "SOLVER_REPS", 2)
    monkeypatch.setattr(C, "SOLVER_J", (20,))
    monkeypatch.setattr(C, "SOLVER_RHOS", (1.0,))
    monkeypatch.setattr(model, "kalman_update", lambda x, g, b, sigma, alpha, eta, taper=None: x * np.nan)
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        res = ev.solver_experiment()
    row = res["rows"][0]
    assert len(row["errs"]) == len(row["spreads"]) == len(row["resid"]) == C.SOLVER_ITERS + 1 and np.isfinite(row["errs"][0]) and np.isnan(row["errs"][1]) and np.isnan(row["errs"][-1])
    assert row["stop_found"] == 0 and row["diverged"] == 0 and np.isfinite(row["err_final"])
