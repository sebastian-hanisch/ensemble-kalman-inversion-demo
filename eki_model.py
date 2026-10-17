"""Kern: das Modell aus Stück 9 (Wärmeleitung und logistische Reaktion), Ensemble-Kalman-Inversion (Update, Tempering, Iteration, Lokalisierung), exakter Posterior im linearen
Fall, MAP und Laplace-Näherung als Referenzen und ein vektorisierter Zufallsschritt-Metropolis-Sampler (Stück 8). Nur numpy, kein Streamlit.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_eki.py` gegen die exakte Kalman-Verstärkung, die Momentenrekursion des Temperings, den exakten Posterior, die Teilraum-Eigenschaft und MCMC.

Verfahren (Iglesias, Law, Stuart 2013; Emerick, Reynolds 2013): Ensemble x_j ~ N(0, tau^2 C) (exponentieller Prior), je Schritt
    x_j <- x_j + C_xg (C_gg + alpha Gamma)^-1 (y + sqrt(alpha) eta_j - G(x_j)),   eta_j ~ N(0, Gamma),   Gamma = sigma^2 I,
mit den Stichproben-Kovarianzen C_xg, C_gg der J Mitglieder. N Schritte mit alpha = N: im linear-gaußschen Fall und J -> unendlich der exakte Posterior (die Summe der 1/alpha
ist 1); N = 1 ist das Ensemble-Kalman-Update; alpha = 1 iteriert ist ein ableitungsfreier Löser (Diskrepanzprinzip als Halt).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import eki_constants as C
from inverse_generator import SplitMix64

_TINY = 1e-300


# ----------------------------------------------------------------------------- Modell (Stück 9)

def heat_matrix(n: int, w: float, k: int) -> np.ndarray:
    """Ein Wärmeleitungsschritt mit reflektierenden Rändern in der Kosinusbasis: Symbol exp(-(w^2 / 2k) (j pi)^2); k Schritte ergeben genau die Unschärfe der Breite w,
    die Zeilensummen sind 1, die Matrix ist symmetrisch."""
    j = np.arange(n)
    c = math.sqrt(2.0 / n) * np.cos(np.pi * (np.arange(n)[:, None] + 0.5) * j[None, :] / n)
    c[:, 0] = 1.0 / math.sqrt(n)
    return (c * np.exp(-(w ** 2 / (2.0 * k)) * (j * math.pi) ** 2)[None, :]) @ c.T


class Model:
    """G(x) = u_K mit u_{k+1} = R(A u_k), R(u) = u + (rho / K) u (1 - u)."""

    def __init__(self, a: np.ndarray, k: int, rho: float):
        self.a = np.asarray(a, dtype=float)
        self.k = int(k)
        self.rho = float(rho)
        self.n = self.a.shape[0]

    def react(self, u):
        return u + (self.rho / self.k) * u * (1.0 - u)

    def dreact(self, u):
        return 1.0 + (self.rho / self.k) * (1.0 - 2.0 * u)

    def forward(self, x):
        u = np.asarray(x, dtype=float)
        for _ in range(self.k):
            u = self.react(self.a @ u)
        return u

    def forward_rows(self, x_rows: np.ndarray) -> np.ndarray:
        """G auf jeder Zeile von x_rows."""
        u = np.asarray(x_rows, dtype=float)
        for _ in range(self.k):
            u = self.react(u @ self.a.T)
        return u

    def jacobian(self, x) -> np.ndarray:
        """Explizite Jacobi-Matrix J = D_K A ... D_1 A mit D_k = diag R'(A u_{k-1})."""
        u = np.asarray(x, dtype=float)
        j = np.eye(self.n)
        for _ in range(self.k):
            v = self.a @ u
            j = (self.dreact(v)[:, None] * self.a) @ j
            u = self.react(v)
        return j

    def total_matrix(self) -> np.ndarray:
        """Die lineare Abbildung A^K (für rho = 0 gleich G)."""
        return np.linalg.matrix_power(self.a, self.k)


# ----------------------------------------------------------------------------- Prior

def exp_kernel(s: np.ndarray, ell: float) -> np.ndarray:
    """Korrelationsmatrix exp(-|s_i - s_j| / ell) (Stück 8)."""
    s = np.asarray(s, dtype=float)
    return np.exp(-np.abs(s[:, None] - s[None, :]) / ell)


def prior_cov(s: np.ndarray) -> np.ndarray:
    """Prior-Kovarianz tau^2 C + 1e-10 I (der Zuschlag hält die Cholesky-Zerlegung stabil)."""
    return C.PRIOR_TAU ** 2 * exp_kernel(s, C.PRIOR_ELL) + 1e-10 * np.eye(len(s))


# ----------------------------------------------------------------------------- Lokalisierung

def gaspari_cohn(d, c: float) -> np.ndarray:
    """Gaspari-Cohn-Funktion mit Halbbreite c (Träger 2c): 1 bei 0, 5/24 bei |d| = c, 0 ab |d| = 2c, stetig."""
    z = np.abs(np.asarray(d, dtype=float)) / c
    out = np.zeros_like(z)
    m1 = z <= 1.0
    m2 = (z > 1.0) & (z < 2.0)
    a = z[m1]
    out[m1] = -0.25 * a ** 5 + 0.5 * a ** 4 + 0.625 * a ** 3 - 5.0 / 3.0 * a ** 2 + 1.0
    b = z[m2]
    out[m2] = b ** 5 / 12.0 - 0.5 * b ** 4 + 0.625 * b ** 3 + 5.0 / 3.0 * b ** 2 - 5.0 * b + 4.0 - 2.0 / (3.0 * b)
    return out


def taper_matrix(s: np.ndarray, c: float):
    """Gaspari-Cohn-Taper über den Abstand der Gitterpunkte (Zustand i, Messung j liegen am selben Gitter); None für c = 0 (keine Lokalisierung)."""
    if c <= 0.0:
        return None
    return gaspari_cohn(s[:, None] - s[None, :], c)


# ----------------------------------------------------------------------------- Ensemble-Kalman-Inversion

def kalman_update(x: np.ndarray, g: np.ndarray, b: np.ndarray, sigma: float, alpha: float, eta: np.ndarray, taper=None) -> np.ndarray:
    """Ein Schritt: x (J, n) Ensemble, g (J, m) = G(x), eta (J, m) Störungen mit Standardabweichung sigma. C_xg und C_gg sind die Stichproben-Kovarianzen (durch J - 1), mit dem Taper
    elementweise multipliziert (beide, sonst passt der Kreuzterm nicht mehr zu C_gg)."""
    j = x.shape[0]
    a = x - x.mean(axis=0)
    bg = g - g.mean(axis=0)
    cxg = a.T @ bg / (j - 1)
    cgg = bg.T @ bg / (j - 1)
    if taper is not None:
        cxg = cxg * taper
        cgg = cgg * taper
    innov = b[None, :] + math.sqrt(alpha) * eta - g
    return x + innov @ np.linalg.solve(cgg + alpha * sigma ** 2 * np.eye(cgg.shape[0]), cxg.T)


@dataclass
class EkiResult:
    ens: np.ndarray            # Ensemble nach dem letzten Schritt (J, n)
    ens_stop: np.ndarray       # Ensemble beim Halt nach dem Diskrepanzprinzip, sonst das letzte
    stop: object               # Iteration des Halts (1, 2, ...) oder None
    means: list                # Ensemblemittel nach 0, 1, ... Schritten
    spreads: list              # mittlere Standardabweichung der Komponenten nach 0, 1, ... Schritten
    resid: list                # |G(Mittel) - b| / (sigma sqrt m) nach 0, 1, ... Schritten
    finite: bool               # kein Schritt hat nichtendliche Werte erzeugt (sonst ist `ens` das Ensemble vor dem fehlgeschlagenen Schritt)
    evals: int = 0             # Vorwärtsläufe: J je Schritt


def run_eki(m: Model, b: np.ndarray, sigma: float, lc: np.ndarray, j: int, steps: int, alpha: float, seed: int, taper=None, disc_factor=None) -> EkiResult:
    """Start-Ensemble aus dem Prior (x = L z, z ~ N(0, I)), danach `steps` Schritte mit der Kopplung alpha. Mit `disc_factor` wird die erste Iteration gemerkt, nach der das
    Residuum des Ensemblemittels höchstens disc_factor mal sigma sqrt(m) beträgt (das Ensemble dort ist `ens_stop`); die Iteration läuft trotzdem weiter."""
    n = m.n
    rng = SplitMix64(seed)
    with np.errstate(all="ignore"):
        x = rng.normal(j * n).reshape(j, n) @ lc.T
        means, spreads, resid = [], [], []
        stop, ens_stop, finite = None, None, True

        def record(xx):
            mu = xx.mean(axis=0)
            means.append(mu)
            spreads.append(float(xx.std(axis=0, ddof=1).mean()))
            resid.append(float(np.linalg.norm(m.forward(mu) - b)) / (sigma * math.sqrt(len(b))))

        record(x)
        for it in range(1, steps + 1):
            g = m.forward_rows(x)
            eta = rng.normal(j * len(b)).reshape(j, len(b)) * sigma
            x_new = kalman_update(x, g, b, sigma, alpha, eta, taper)
            if not np.all(np.isfinite(x_new)):
                finite = False
                break
            x = x_new
            record(x)
            if disc_factor is not None and stop is None and resid[-1] <= disc_factor:
                stop, ens_stop = it, x.copy()
    return EkiResult(ens=x, ens_stop=x if ens_stop is None else ens_stop, stop=stop, means=means, spreads=spreads, resid=resid, finite=finite, evals=j * steps)


def ensemble_stats(ens: np.ndarray, mean_ref: np.ndarray, sd_ref: np.ndarray) -> tuple[float, float, float, float]:
    """(Mittelwertabstand in sd der Referenz (RMS über die Komponenten), sd-Verhältnis Ensemble / Referenz: Median, kleinster, größter Wert über die Komponenten)."""
    mean = ens.mean(axis=0)
    sd = ens.std(axis=0, ddof=1)
    ratio = sd / sd_ref
    return float(np.sqrt(np.mean(((mean - mean_ref) / sd_ref) ** 2))), float(np.median(ratio)), float(ratio.min()), float(ratio.max())


# ----------------------------------------------------------------------------- Referenzen: exakt, MAP, Laplace, MCMC

def linear_posterior(a_total: np.ndarray, prior_covariance: np.ndarray, sigma: float, b: np.ndarray):
    """Exakter Posterior N(m, G) für b = A x + e, e ~ N(0, sigma^2 I), x ~ N(0, P) in Datenraumformeln: (Mittel, Kovarianz)."""
    k = a_total @ prior_covariance @ a_total.T + sigma ** 2 * np.eye(a_total.shape[0])
    gain = prior_covariance @ a_total.T
    mean = gain @ np.linalg.solve(k, b)
    cov = prior_covariance - gain @ np.linalg.solve(k, gain.T)
    return mean, 0.5 * (cov + cov.T)


def identification_eigenvalues(a_total: np.ndarray, prior_covariance: np.ndarray, sigma: float) -> np.ndarray:
    """Eigenwerte (absteigend) der prior-gewichteten Hesse-Matrix R^T A^T A R / sigma^2 mit R R^T = P: lambda_i > 1 heißt, die Daten bestimmen die Richtung stärker als der Prior
    (Stück 8)."""
    ev, evec = np.linalg.eigh(0.5 * (prior_covariance + prior_covariance.T))
    r = evec * np.sqrt(np.maximum(ev, 0.0))
    h = r.T @ a_total.T @ a_total @ r / sigma ** 2
    return np.sort(np.linalg.eigvalsh(0.5 * (h + h.T)))[::-1]


def map_estimate(m: Model, b: np.ndarray, sigma: float, lc: np.ndarray) -> tuple[np.ndarray, int, int]:
    """MAP von |G(x) - b|^2 / (2 sigma^2) + x^T P^-1 x / 2: Levenberg-Marquardt in Whitening-Koordinaten z (x = L z, Prior N(0, I)). Gibt (MAP, Iterationen, Jacobi-Matrizen) zurück."""
    n = m.n
    z = np.zeros(n)
    mu = C.MU_START

    def cost(zz):
        r = m.forward(lc @ zz) - b
        return 0.5 * float(r @ r) / sigma ** 2 + 0.5 * float(zz @ zz)

    f = cost(z)
    iterations = jacs = 0
    for _ in range(C.LM_MAXIT):
        r = m.forward(lc @ z) - b
        j = m.jacobian(lc @ z) @ lc
        jacs += 1
        g = j.T @ r / sigma ** 2 + z
        if np.linalg.norm(g) <= C.LM_TOL * (1.0 + np.linalg.norm(z)):
            break
        h = j.T @ j / sigma ** 2 + np.eye(n)
        for _ in range(C.LM_TRIES):
            with np.errstate(all="ignore"):
                zn = z - np.linalg.solve(h + mu * np.eye(n), g)
                fn = cost(zn)
            if fn < f:
                z, f, mu = zn, fn, max(mu * C.MU_DOWN, C.MU_MIN)
                break
            mu *= C.MU_UP
        iterations += 1
    return lc @ z, iterations, jacs


def laplace_covariance(m: Model, x_map: np.ndarray, sigma: float, prior_covariance: np.ndarray) -> np.ndarray:
    """Kovarianz der Laplace-Näherung N(x_map, H^-1) mit der Gauss-Newton-Hesse-Matrix H = J^T J / sigma^2 + P^-1 am MAP."""
    j = m.jacobian(x_map)
    cov = np.linalg.inv(j.T @ j / sigma ** 2 + np.linalg.inv(prior_covariance))
    return 0.5 * (cov + cov.T)


def neg_log_post(m: Model, x_rows: np.ndarray, b: np.ndarray, sigma: float, prec: np.ndarray) -> np.ndarray:
    """Negative Log-Posterior |G(x) - b|^2 / (2 sigma^2) + x^T P^-1 x / 2 für jede Zeile von x_rows."""
    x_rows = np.atleast_2d(x_rows)
    r = m.forward_rows(x_rows) - b[None, :]
    return 0.5 * np.sum(r ** 2, axis=1) / sigma ** 2 + 0.5 * np.einsum("ij,jk,ik->i", x_rows, prec, x_rows)


def rwm(m: Model, b: np.ndarray, sigma: float, prec: np.ndarray, x0: np.ndarray, steps: int, seed: int, step: float, chol: np.ndarray, fvec: np.ndarray) -> dict:
    """Zufallsschritt-Metropolis, vektorisiert über die Ketten (Zeilen von x0): x' = x + step L z, Annahme exp(U(x) - U(x')). Gibt zurück: sum1, sum2 (Summen von x und x^2
    nach dem Verwerfen, je Kette), count, acc (Annahmerate je Kette), traces (steps, chains) des Funktionals fvec @ x."""
    x = np.array(np.atleast_2d(x0), dtype=float)
    chains, n = x.shape
    rng = SplitMix64(seed)
    burn = int(C.BURN_FRACTION * steps)
    cur = -neg_log_post(m, x, b, sigma, prec)
    s1, s2 = np.zeros((chains, n)), np.zeros((chains, n))
    acc = np.zeros(chains)
    traces = np.empty((steps, chains))
    kept, done = 0, 0
    while done < steps:
        blk = min(C.BLOCK, steps - done)
        z = rng.normal(blk * chains * n).reshape(blk, chains, n)
        u = np.log(np.maximum(rng.uniform(blk * chains).reshape(blk, chains), _TINY))
        for i in range(blk):
            prop = x + step * (z[i] @ chol.T)
            new = -neg_log_post(m, prop, b, sigma, prec)
            ok = u[i] < new - cur
            x = np.where(ok[:, None], prop, x)
            cur = np.where(ok, new, cur)
            acc += ok
            traces[done + i] = x @ fvec
            if done + i >= burn:
                s1 += x
                s2 += x ** 2
                kept += 1
        done += blk
    return {"sum1": s1, "sum2": s2, "count": kept, "acc": acc / steps, "traces": traces}


def rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin R-Dach für Ketten der Form (chains, steps)."""
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    w = float(chains.var(axis=1, ddof=1).mean())
    bvar = n * float(chains.mean(axis=1).var(ddof=1))
    if w <= 0.0:
        return math.inf
    return math.sqrt(((n - 1) / n * w + bvar / n) / w)
