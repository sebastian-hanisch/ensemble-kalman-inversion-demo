"""Orakel (DEMO-PLAYBOOK §12): exakte Kalman-Verstärkung, Momentenrekursion des Temperings, Iteration gleich Tikhonov (Momente), Ensemble gegen den exakten Posterior (Konvergenz mit J),
Teilraum-Eigenschaft, Stichproben-Kovarianzen gegen np.cov, Lokalisierung (positiv semidefinit, Grenzfall), MCMC gegen exakten Posterior und Laplace-Näherung. Jeder Test unter etwa 10 s."""

import math

import numpy as np
import pytest

import eki_evaluation as ev
import eki_model as m
from inverse_generator import SplitMix64, grid


def moment_update(mean, cov, a, b, sigma, alpha):
    """Ein Kalman-Schritt der Momente (exakte Kovarianzen statt Stichproben) mit der Kopplung alpha: Mittel und Kovarianz des Zustands."""
    cxg = cov @ a.T
    k = cxg @ np.linalg.inv(a @ cov @ a.T + alpha * sigma ** 2 * np.eye(a.shape[0]))
    return mean + k @ (b - a @ mean), cov - k @ a @ cov


def _linear_problem(noise=1e-2, n=48):
    inv, x, b, sigma, pcov, lc = ev.problem(0.0, noise, n=n)
    return inv, x, b, sigma, pcov, lc


# ----------------------------------------------------------------------------- Exakte Momente

def test_exact_kalman_gain_gives_the_posterior_of_the_information_form():
    inv, x, b, sigma, pcov, lc = _linear_problem()
    a = inv.total_matrix()
    mean, cov = moment_update(np.zeros(48), pcov, a, b, sigma, 1.0)
    info = np.linalg.inv(a.T @ a / sigma ** 2 + np.linalg.inv(pcov))                # zweite Route: Informationsform
    assert cov == pytest.approx(info, rel=1e-6, abs=1e-10) and mean == pytest.approx(info @ a.T @ b / sigma ** 2, abs=1e-6)
    ref_mean, ref_cov = m.linear_posterior(a, pcov, sigma, b)
    assert mean == pytest.approx(ref_mean, abs=1e-8) and cov == pytest.approx(ref_cov, abs=1e-10)


@pytest.mark.parametrize("steps", [1, 2, 4, 16])
def test_tempering_with_alpha_n_and_exact_moments_is_the_exact_posterior(steps):
    inv, x, b, sigma, pcov, lc = _linear_problem()
    a = inv.total_matrix()
    mean, cov = np.zeros(48), pcov.copy()
    for _ in range(steps):
        mean, cov = moment_update(mean, cov, a, b, sigma, float(steps))
    ref_mean, ref_cov = m.linear_posterior(a, pcov, sigma, b)
    assert mean == pytest.approx(ref_mean, abs=1e-6) and cov == pytest.approx(ref_cov, abs=1e-8)


def test_iterating_with_alpha_one_and_exact_moments_is_tikhonov_with_shrinking_lambda(mini):
    a, x, b = mini
    sigma, tau = 0.1, 1.0
    mean, cov = np.zeros(3), tau ** 2 * np.eye(3)
    for k in range(1, 6):
        mean, cov = moment_update(mean, cov, a, b, sigma, 1.0)
        lam = sigma / (tau * math.sqrt(k))                                       # k Kopien der Daten: lambda² = sigma² / (k tau²)
        assert mean == pytest.approx(np.linalg.solve(a.T @ a + lam ** 2 * np.eye(3), a.T @ b), abs=1e-9)
    assert mean[0] != pytest.approx(1.153465, abs=1e-3)                          # nicht der Posterior (der gilt nur für eine Kopie)


# ----------------------------------------------------------------------------- Ensemble gegen den exakten Posterior

def test_ensemble_converges_to_the_exact_posterior_like_one_over_sqrt_j():
    inv, x, b, sigma, pcov, lc = _linear_problem()
    mean, cov = m.linear_posterior(inv.total_matrix(), pcov, sigma, b)
    sd = np.sqrt(np.diag(cov))

    def stats(j, reps):
        res = [m.ensemble_stats(m.run_eki(inv, b, sigma, lc, j, 1, 1.0, 100 + r).ens, mean, sd) for r in range(reps)]
        return float(np.median([q[0] for q in res])), float(np.median([q[1] for q in res]))

    d400, r400 = stats(400, 10)
    d1600, r1600 = stats(1600, 10)
    d6400, r6400 = stats(6400, 5)
    assert 1.5 < d400 / d1600 < 2.8 and 1.4 < d1600 / d6400 < 2.8 and d6400 < 0.12
    assert 0.93 < r400 < 1.0 and 0.97 < r1600 < 1.01 and 0.99 < r6400 < 1.01
    d10, r10 = stats(10, 10)
    assert r10 < 0.1 and d10 > 1.5                                               # J = 10: die Streuung ist zusammengebrochen, das Mittel liegt weit daneben


def test_tempering_with_alpha_n_loses_spread_but_not_the_mean_for_a_finite_ensemble():
    inv, x, b, sigma, pcov, lc = _linear_problem()
    mean, cov = m.linear_posterior(inv.total_matrix(), pcov, sigma, b)
    sd = np.sqrt(np.diag(cov))
    one = [m.ensemble_stats(m.run_eki(inv, b, sigma, lc, 100, 1, 1.0, 300 + r).ens, mean, sd) for r in range(10)]
    many = [m.ensemble_stats(m.run_eki(inv, b, sigma, lc, 100, 16, 16.0, 300 + r).ens, mean, sd) for r in range(10)]
    assert np.median([q[1] for q in many]) < np.median([q[1] for q in one]) - 0.1 and abs(np.median([q[0] for q in many]) - np.median([q[0] for q in one])) < 0.15


# ----------------------------------------------------------------------------- Teilraum-Eigenschaft und Stichproben-Kovarianzen

def test_update_lies_in_the_span_of_the_anomalies_only_without_a_taper():
    n, j = 48, 10
    rng = SplitMix64(4)
    inv, x, b, sigma, pcov, lc = _linear_problem()
    xe = rng.normal(j * n).reshape(j, n) @ lc.T
    g = inv.forward_rows(xe)
    eta = rng.normal(j * n).reshape(j, n) * sigma
    inc = m.kalman_update(xe, g, b, sigma, 1.0, eta) - xe
    anomalies = xe - xe.mean(axis=0)
    q, _ = np.linalg.qr(anomalies.T)                                             # Orthonormalbasis des Aufspanns (Rang j - 1)
    resid = inc - (inc @ q) @ q.T
    assert np.linalg.matrix_rank(anomalies) == j - 1 and np.linalg.norm(resid) < 1e-9 * np.linalg.norm(inc) and np.linalg.matrix_rank(inc, tol=1e-9 * np.linalg.norm(inc)) <= j
    inc_loc = m.kalman_update(xe, g, b, sigma, 1.0, eta, m.taper_matrix(grid(n), 0.2)) - xe
    resid_loc = inc_loc - (inc_loc @ q) @ q.T
    assert np.linalg.norm(resid_loc) > 1e-3 * np.linalg.norm(inc_loc)             # die Lokalisierung verlässt den Aufspann


def test_sample_covariances_are_np_cov():
    n, j = 12, 7
    rng = SplitMix64(8)
    inv, x, b, sigma, pcov, lc = _linear_problem(n=n)
    xe = rng.normal(j * n).reshape(j, n) @ lc.T
    g = inv.forward_rows(xe)
    eta = np.zeros((j, n))
    cov = np.cov(np.hstack([xe, g]).T)
    cxg, cgg = cov[:n, n:], cov[n:, n:]
    expected = xe + (b[None, :] - g) @ np.linalg.solve(cgg + sigma ** 2 * np.eye(n), cxg.T)
    assert m.kalman_update(xe, g, b, sigma, 1.0, eta) == pytest.approx(expected, abs=1e-8)


# ----------------------------------------------------------------------------- Lokalisierung

def test_large_halfwidth_localization_is_no_localization_and_small_one_decouples():
    n, j = 24, 30
    rng = SplitMix64(12)
    inv, x, b, sigma, pcov, lc = _linear_problem(n=n)
    xe = rng.normal(j * n).reshape(j, n) @ lc.T
    g = inv.forward_rows(xe)
    eta = rng.normal(j * n).reshape(j, n) * sigma
    plain = m.kalman_update(xe, g, b, sigma, 1.0, eta)
    wide = m.kalman_update(xe, g, b, sigma, 1.0, eta, m.taper_matrix(grid(n), 1e4))
    assert wide == pytest.approx(plain, rel=1e-6, abs=1e-8)
    tiny = m.kalman_update(xe, g, b, sigma, 1.0, eta, m.taper_matrix(grid(n), 1e-3))              # Taper = Einheitsmatrix: nur die Diagonalen bleiben
    cov = np.cov(np.hstack([xe, g]).T)
    diag_gain = np.diag(np.diag(cov[:n, n:]) / (np.diag(cov[n:, n:]) + sigma ** 2))
    assert tiny == pytest.approx(xe + (b[None, :] + eta - g) @ diag_gain.T, abs=1e-8)


def test_localization_rescues_a_small_ensemble():
    inv, x, b, sigma, pcov, lc = _linear_problem()
    mean, cov = m.linear_posterior(inv.total_matrix(), pcov, sigma, b)
    sd = np.sqrt(np.diag(cov))
    s = grid(48)
    plain = [m.ensemble_stats(m.run_eki(inv, b, sigma, lc, 10, 1, 1.0, 500 + r).ens, mean, sd) for r in range(10)]
    loc = [m.ensemble_stats(m.run_eki(inv, b, sigma, lc, 10, 1, 1.0, 500 + r, m.taper_matrix(s, 0.2)).ens, mean, sd) for r in range(10)]
    tiny = [m.ensemble_stats(m.run_eki(inv, b, sigma, lc, 10, 1, 1.0, 500 + r, m.taper_matrix(s, 0.05)).ens, mean, sd) for r in range(10)]
    assert np.median([q[1] for q in plain]) < 0.1 and 0.7 < np.median([q[1] for q in loc]) < 1.05 and np.median([q[0] for q in loc]) < 0.5 * np.median([q[0] for q in plain])
    assert np.median([q[1] for q in tiny]) > 1.25                                 # zu kleine Halbbreite: zu viel Streuung


# ----------------------------------------------------------------------------- MCMC-Referenz

def _chain_moments(inv, b, sigma, pcov, x_map, cov, chains, steps, seed):
    n = len(x_map)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(n))
    x0 = x_map[None, :] + 2.0 * SplitMix64(seed).normal(chains * n).reshape(chains, n) @ chol.T
    r = m.rwm(inv, b, sigma, np.linalg.inv(pcov), x0, steps, seed + 1, 2.38 / math.sqrt(n), chol, np.full(n, 1.0 / n))
    total = r["count"] * chains
    mean = r["sum1"].sum(axis=0) / total
    return mean, np.sqrt(np.maximum(r["sum2"].sum(axis=0) / total - mean ** 2, 0.0)), r


def test_mcmc_reference_reproduces_the_exact_linear_posterior():
    n = 12
    inv, x, b, sigma, pcov, lc = _linear_problem(n=n)
    mean, cov = m.linear_posterior(inv.total_matrix(), pcov, sigma, b)
    x_map = m.map_estimate(inv, b, sigma, lc)[0]
    assert x_map == pytest.approx(mean, abs=1e-6)
    cmean, csd, r = _chain_moments(inv, b, sigma, pcov, x_map, cov, 8, 20000, 7)
    ref = np.sqrt(np.diag(cov))
    assert np.max(np.abs(cmean - mean) / ref) < 0.2 and np.max(np.abs(csd / ref - 1.0)) < 0.08 and float(r["acc"].mean()) == pytest.approx(0.25, abs=0.08)


def test_laplace_approximation_matches_mcmc_for_a_moderately_nonlinear_posterior():
    n = 12
    inv, x, b, sigma, pcov, lc = ev.problem(1.0, 1e-2, n=n)
    x_map = m.map_estimate(inv, b, sigma, lc)[0]
    cov = m.laplace_covariance(inv, x_map, sigma, pcov)
    cmean, csd, r = _chain_moments(inv, b, sigma, pcov, x_map, cov, 8, 20000, 11)
    ratio = np.sqrt(np.diag(cov)) / csd
    assert np.all(np.abs(ratio - 1.0) < 0.06) and np.max(np.abs(cmean - x_map) / csd) < 0.3 and m.rhat(r["traces"][4000:].T) < 1.05
