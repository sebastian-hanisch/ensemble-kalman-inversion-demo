"""Kernroutinen an der 3 x 3-Mini-Instanz und am handgebauten Ensemble von Hand (DEMO-PLAYBOOK §1b); jede Einheit einzeln."""

import math

import numpy as np
import pytest

import eki_constants as C
import eki_model as m
from inverse_generator import SplitMix64, grid

KB = np.array([1.205575, 1.491289, 1.205575])          # K b des handgebauten Ensembles für alpha = 1


# ----------------------------------------------------------------------------- Modell und Prior

def test_heat_matrix_n2_by_hand_and_semigroup():
    s = math.exp(-(0.1 ** 2 / 4.0) * math.pi ** 2)                      # K = 2, w = 0,1: Symbol von j = 1: exp(-(w²/2K) π²)
    assert m.heat_matrix(2, 0.1, 2) == pytest.approx(0.5 * np.array([[1 + s, 1 - s], [1 - s, 1 + s]]))
    a = m.heat_matrix(12, 0.05, 4)
    assert a == pytest.approx(a.T) and a.sum(axis=1) == pytest.approx(np.ones(12)) and np.linalg.matrix_power(a, 4) == pytest.approx(m.heat_matrix(12, 0.05, 1), abs=1e-12)


def test_model_forward_jacobian_and_total_matrix_by_hand(mini, mini_model, mini_model_nl):
    a, x, b = mini
    assert mini_model.forward(x) == pytest.approx(a @ x) and mini_model_nl.forward(x) == pytest.approx([0.91, 0.84, 0.91])
    assert mini_model_nl.react(1.3) == pytest.approx(0.91) and mini_model_nl.dreact(1.3) == pytest.approx(-0.6) and mini_model_nl.dreact(1.0) == 0.0
    assert mini_model_nl.jacobian(x) == pytest.approx(np.array([[-0.36, -0.18, -0.06], [-0.24, -0.32, -0.24], [-0.06, -0.18, -0.36]]))
    assert mini_model.jacobian(x) == pytest.approx(a)
    assert m.Model(a, 3, 0.0).total_matrix() == pytest.approx(np.linalg.matrix_power(a, 3)) and mini_model.n == 3
    rows = np.vstack([x, 0.5 * x, np.zeros(3)])
    out = mini_model_nl.forward_rows(rows)
    assert all(out[i] == pytest.approx(mini_model_nl.forward(rows[i])) for i in range(3))
    two = m.Model(a, 2, 2.0)                                            # zwei Schritte, dieselbe Reaktion 2u - u² je Schritt
    assert two.forward(x) == pytest.approx([0.987679, 0.986076, 0.987679], abs=1e-6)
    h = 1e-6
    fd = np.column_stack([(two.forward(x + h * e) - two.forward(x - h * e)) / (2 * h) for e in np.eye(3)])
    assert two.jacobian(x) == pytest.approx(fd, abs=1e-8)


def test_exp_kernel_and_prior_cov_by_hand():
    k = m.exp_kernel(np.array([0.0, 1.0]), 2.0)
    assert k == pytest.approx(np.array([[1.0, math.exp(-0.5)], [math.exp(-0.5), 1.0]]))
    p = m.prior_cov(np.array([0.0, 1.0]))
    assert p[0, 0] == pytest.approx(C.PRIOR_TAU ** 2 + 1e-10) and p[0, 1] == pytest.approx(C.PRIOR_TAU ** 2 * math.exp(-1.0 / C.PRIOR_ELL))


# ----------------------------------------------------------------------------- Lokalisierung

def test_gaspari_cohn_by_hand():
    assert m.gaspari_cohn([0.0, 1.0, 2.0, 3.0], 1.0) == pytest.approx([1.0, 5.0 / 24.0, 0.0, 0.0])
    assert m.gaspari_cohn(0.5, 1.0) == pytest.approx(0.6848958, abs=1e-6) and m.gaspari_cohn(1.5, 1.0) == pytest.approx(0.0164931, abs=1e-6)
    assert m.gaspari_cohn([-0.5, 0.5], 1.0) == pytest.approx([0.6848958, 0.6848958], abs=1e-6)
    assert m.gaspari_cohn(0.3, 0.6) == pytest.approx(m.gaspari_cohn(0.5, 1.0))                      # Halbbreite skaliert den Abstand
    left, right = m.gaspari_cohn(1.0 - 1e-9, 1.0), m.gaspari_cohn(1.0 + 1e-9, 1.0)
    assert left == pytest.approx(right, abs=1e-8) and m.gaspari_cohn(2.0 - 1e-9, 1.0) < 1e-8        # stetig bei c und 2c


def test_taper_matrix_by_hand_and_positive_semidefinite():
    assert m.taper_matrix(grid(4), 0.0) is None
    t = m.taper_matrix(grid(4), 0.25)
    assert t == pytest.approx(t.T) and np.diag(t) == pytest.approx(np.ones(4)) and t[0, 1] == pytest.approx(5.0 / 24.0) and t[0, 2] == pytest.approx(0.0, abs=1e-12)
    for c in (0.05, 0.1, 0.2, 0.4):
        assert np.linalg.eigvalsh(m.taper_matrix(grid(48), c)).min() > -1e-12


# ----------------------------------------------------------------------------- Ensemble-Kalman-Schritt

def test_kalman_update_by_hand(mini, mini_model, mini_ensemble):
    a, x, b = mini
    g = mini_model.forward_rows(mini_ensemble)
    zero = np.zeros((6, 3))
    assert m.kalman_update(mini_ensemble, g, b, 0.1, 1.0, zero).mean(axis=0) == pytest.approx(KB, abs=2e-6)
    assert m.kalman_update(mini_ensemble, g, b, 0.1, 4.0, zero).mean(axis=0) == pytest.approx([1.181818, 1.272727, 1.181818], abs=2e-6)
    assert m.kalman_update(mini_ensemble, g, b, 0.1, 1.0, zero, np.eye(3)).mean(axis=0) == pytest.approx([1.608247, 1.534247, 1.608247], abs=2e-6)
    gain = 0.4 * a @ np.linalg.inv(0.4 * a @ a + 0.01 * np.eye(3))
    assert sorted(np.linalg.eigvalsh(gain)) == pytest.approx([0.97561, 1.81818, 2.857143], abs=1e-5)


def test_kalman_update_perturbation_enters_with_sqrt_alpha(mini, mini_model, mini_ensemble):
    a, x, b = mini
    g = mini_model.forward_rows(mini_ensemble)
    eta = np.full((6, 3), 0.1)
    gain4 = 0.4 * a @ np.linalg.inv(0.4 * a @ a + 4.0 * 0.01 * np.eye(3))
    out = m.kalman_update(mini_ensemble, g, b, 0.1, 4.0, eta)
    assert out.mean(axis=0) == pytest.approx(gain4 @ (b + 2.0 * np.full(3, 0.1)), abs=1e-9)           # sqrt(4) = 2
    out1 = m.kalman_update(mini_ensemble, g, b, 0.1, 1.0, eta)
    assert out1.mean(axis=0) == pytest.approx(0.4 * a @ np.linalg.inv(0.4 * a @ a + 0.01 * np.eye(3)) @ (b + 0.1), abs=1e-9)


def test_kalman_update_member_by_hand(mini, mini_model, mini_ensemble):
    a, x, b = mini
    g = mini_model.forward_rows(mini_ensemble)
    out = m.kalman_update(mini_ensemble, g, b, 0.1, 1.0, np.zeros((6, 3)))
    gain = 0.4 * a @ np.linalg.inv(0.4 * a @ a + 0.01 * np.eye(3))
    assert out[0] == pytest.approx(np.array([1.0, 0.0, 0.0]) + gain @ (b - a[:, 0]), abs=1e-9) and out[3] == pytest.approx(np.array([-1.0, 0.0, 0.0]) + gain @ (b + a[:, 0]), abs=1e-9)


# ----------------------------------------------------------------------------- run_eki

def test_run_eki_bookkeeping_and_start_ensemble(mini, mini_model_nl):
    a, x, b = mini
    lc = np.linalg.cholesky(np.eye(3))
    r = m.run_eki(mini_model_nl, b, 0.1, lc, 6, 3, 3.0, 5)
    assert r.ens.shape == (6, 3) and len(r.means) == len(r.spreads) == len(r.resid) == 4 and r.evals == 18 and r.finite and r.stop is None and r.ens_stop is r.ens
    z0 = SplitMix64(5).normal(18).reshape(6, 3)
    assert r.means[0] == pytest.approx(z0.mean(axis=0)) and r.spreads[0] == pytest.approx(float(z0.std(axis=0, ddof=1).mean()))
    assert r.resid[0] == pytest.approx(float(np.linalg.norm(mini_model_nl.forward(z0.mean(axis=0)) - b)) / (0.1 * math.sqrt(3)))
    again = m.run_eki(mini_model_nl, b, 0.1, lc, 6, 3, 3.0, 5)
    assert np.array_equal(r.ens, again.ens) and not np.array_equal(r.ens, m.run_eki(mini_model_nl, b, 0.1, lc, 6, 3, 3.0, 6).ens)
    shifted = m.run_eki(mini_model_nl, b, 0.1, 2.0 * lc, 6, 0, 1.0, 5)                       # Start-Ensemble ist L z
    assert shifted.ens == pytest.approx(2.0 * z0) and shifted.evals == 0 and len(shifted.means) == 1


def test_run_eki_one_step_equals_one_update_of_the_start_ensemble(mini, mini_model):
    a, x, b = mini
    lc = np.eye(3)
    r = m.run_eki(mini_model, b, 0.1, lc, 6, 1, 1.0, 9)
    rng = SplitMix64(9)
    x0 = rng.normal(18).reshape(6, 3)
    eta = rng.normal(18).reshape(6, 3) * 0.1
    assert r.ens == pytest.approx(m.kalman_update(x0, mini_model.forward_rows(x0), b, 0.1, 1.0, eta))


def test_run_eki_discrepancy_stop_keeps_the_ensemble_at_the_stop(mini, mini_model):
    a, x, b = mini
    lc = np.eye(3)
    one = m.run_eki(mini_model, b, 0.1, lc, 6, 1, 1.0, 9)
    big = m.run_eki(mini_model, b, 0.1, lc, 6, 3, 1.0, 9, None, 1e9)
    assert big.stop == 1 and big.ens_stop == pytest.approx(one.ens) and not np.array_equal(big.ens, big.ens_stop)
    never = m.run_eki(mini_model, b, 0.1, lc, 6, 3, 1.0, 9, None, 1e-9)
    assert never.stop is None and never.ens_stop is never.ens
    first = next(i for i, v in enumerate(big.resid) if i >= 1 and v <= 1.5)
    mid = m.run_eki(mini_model, b, 0.1, lc, 6, 3, 1.0, 9, None, big.resid[2] + 1e-12)
    assert mid.stop is not None and mid.stop <= 2 and first >= 1


def test_run_eki_stops_at_a_nonfinite_ensemble_and_keeps_the_last_finite_one(mini, mini_model, monkeypatch):
    a, x, b = mini
    calls = []

    def bad(x_, g, b_, sigma, alpha, eta, taper=None):
        calls.append(1)
        return x_ * np.nan

    monkeypatch.setattr(m, "kalman_update", bad)
    r = m.run_eki(mini_model, b, 0.1, np.eye(3), 6, 5, 1.0, 9)
    assert not r.finite and len(calls) == 1 and np.all(np.isfinite(r.ens)) and len(r.means) == 1 and r.ens == pytest.approx(SplitMix64(9).normal(18).reshape(6, 3))


def test_ensemble_stats_by_hand():
    ens = np.array([[0.0, 0.0], [2.0, 2.0]])
    dm, med, lo, hi = m.ensemble_stats(ens, np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    assert dm == pytest.approx(math.sqrt(0.5)) and med == pytest.approx(0.5 * (math.sqrt(2.0) + math.sqrt(2.0) / 2.0)) and lo == pytest.approx(math.sqrt(2.0) / 2.0) and hi == pytest.approx(math.sqrt(2.0))


# ----------------------------------------------------------------------------- Referenzen

def test_linear_posterior_and_identification_by_hand(mini):
    a, x, b = mini
    mean, cov = m.linear_posterior(a, np.eye(3), 0.1, b)
    assert mean == pytest.approx([1.153465, 1.653465, 1.153465], abs=2e-6)
    assert cov[0, 0] == pytest.approx(0.105864, abs=2e-6) and cov[1, 1] == pytest.approx(0.336634, abs=2e-6) and cov[0, 2] == pytest.approx(0.067403, abs=2e-6)
    assert sorted(np.linalg.eigvalsh(cov)) == pytest.approx([1.0 / 101.0, 1.0 / 26.0, 0.5])
    assert m.identification_eigenvalues(a, np.eye(3), 0.2) == pytest.approx([25.0, 6.25, 0.25])
    assert m.identification_eigenvalues(a, 4.0 * np.eye(3), 0.2) == pytest.approx([100.0, 25.0, 1.0])


def test_map_is_the_posterior_mean_for_a_linear_model_and_stationary_for_a_nonlinear_one(mini, mini_model, mini_model_nl):
    a, x, b = mini
    x_map, iters, jacs = m.map_estimate(mini_model, b, 0.1, np.eye(3))
    assert x_map == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-6) and jacs >= 1 and iters >= 1
    bn = mini_model_nl.forward(x)
    x_nl, iters, jacs = m.map_estimate(mini_model_nl, bn, 0.1, np.eye(3))
    h = 1e-6

    def cost(v):
        r = mini_model_nl.forward(v) - bn
        return 0.5 * float(r @ r) / 0.01 + 0.5 * float(v @ v)

    fd = np.array([(cost(x_nl + h * e) - cost(x_nl - h * e)) / (2 * h) for e in np.eye(3)])
    assert np.linalg.norm(fd) < 1e-4 and jacs >= iters


def test_map_with_a_correlated_prior_uses_the_whitening_factor(mini, mini_model):
    a, x, b = mini
    p = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])
    x_map = m.map_estimate(mini_model, b, 0.1, np.linalg.cholesky(p))[0]
    assert x_map == pytest.approx(m.linear_posterior(a, p, 0.1, b)[0], abs=1e-6)


def test_laplace_covariance_by_hand(mini, mini_model, mini_model_nl):
    a, x, b = mini
    cov = m.laplace_covariance(mini_model, np.zeros(3), 0.1, np.eye(3))
    assert cov[0, 0] == pytest.approx(0.105864, abs=2e-6) and cov[1, 1] == pytest.approx(0.336634, abs=2e-6) and cov[0, 2] == pytest.approx(0.067403, abs=2e-6)
    j = mini_model_nl.jacobian(x)
    assert m.laplace_covariance(mini_model_nl, x, 0.1, 4.0 * np.eye(3)) == pytest.approx(np.linalg.inv(j.T @ j / 0.01 + np.eye(3) / 4.0))


def test_neg_log_post_by_hand(mini, mini_model, mini_model_nl):
    a, x, b = mini
    both = np.vstack([x, np.zeros(3)])
    val = m.neg_log_post(mini_model, both, b, 0.1, np.eye(3))
    assert val[0] == pytest.approx(3.0) and val[1] == pytest.approx(0.5 * float(b @ b) / 0.01)
    assert m.neg_log_post(mini_model_nl, x, b, 0.1, 2.0 * np.eye(3))[0] == pytest.approx(0.6178 / 0.02 + 6.0)


def test_rwm_mechanics_and_block_handling(mini, mini_model):
    a, x, b = mini
    p = np.eye(3)
    chol = np.linalg.cholesky(np.linalg.inv(a @ a / 0.01 + p))
    f = np.full(3, 1.0 / 3.0)
    r = m.rwm(mini_model, b, 0.1, p, np.tile(x, (4, 1)), 10, 3, 0.5, chol, f)
    assert r["traces"].shape == (10, 4) and r["count"] == 8 and r["sum1"].shape == (4, 3) and np.all((r["acc"] >= 0) & (r["acc"] <= 1)) and np.all(r["sum2"] >= 0)
    r2 = m.rwm(mini_model, b, 0.1, p, np.tile(x, (4, 1)), 10, 3, 0.5, chol, f)
    assert np.array_equal(r["traces"], r2["traces"]) and not np.array_equal(r["traces"], m.rwm(mini_model, b, 0.1, p, np.tile(x, (4, 1)), 10, 4, 0.5, chol, f)["traces"])
    tiny = m.rwm(mini_model, b, 0.1, p, np.tile(x, (4, 1)), 200, 3, 1e-9, chol, np.ones(3))
    assert tiny["acc"].min() > 0.99 and tiny["count"] == 160 and tiny["sum1"] == pytest.approx(np.tile(160 * x, (4, 1))) and tiny["traces"][:, 0] == pytest.approx(np.full(200, 4.0))
    assert m.rwm(mini_model, b, 0.1, p, np.tile(x, (4, 1)), 200, 3, 1e3, chol, f)["acc"].max() < 0.05
    long = m.rwm(mini_model, b, 0.1, p, np.tile(x, (2, 1)), 700, 3, 0.1, np.eye(3), f)                 # 700 ist kein Vielfaches des Blocks 500
    assert long["traces"].shape == (700, 2) and long["count"] == 560 and np.all(np.isfinite(long["traces"]))


def test_rhat_by_hand():
    assert m.rhat(np.array([[0.0, 1.0], [0.0, 1.0]])) == pytest.approx(math.sqrt(0.5))
    assert m.rhat(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])) == pytest.approx(math.sqrt(1.0 + 1.0 / 6.0))
    assert math.isinf(m.rhat(np.array([[0.0, 0.0], [1.0, 1.0]])))


def test_map_converges_in_few_levenberg_marquardt_iterations(mini, mini_model, mini_model_nl):
    a, x, b = mini
    _, iters, jacs = m.map_estimate(mini_model, b, 0.1, np.eye(3))
    assert iters <= 4 and jacs == iters + 1                      # quadratisches Problem: ein fast voller Newton-Schritt je Iteration, dazu die Abbruchprüfung
    bn = mini_model_nl.forward(x)
    _, iters_nl, jacs_nl = m.map_estimate(mini_model_nl, bn, 0.1, np.eye(3))
    assert iters_nl < 20 and jacs_nl == iters_nl + 1 and iters_nl > iters
