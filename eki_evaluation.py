"""Auswertung: Formatierung, Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import eki_constants as C
import eki_model as model
from inverse_generator import SplitMix64, grid, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def num_text(x: float) -> str:
    """Zahl mit drei gültigen Stellen, ab 1000 und unter 0,01 als Zehnerpotenz."""
    if not math.isfinite(x):
        return "unendlich"
    if abs(x) >= 1000.0 or (x != 0.0 and abs(x) < 0.01):
        return sci(x, 1)
    return de(x, 1 if abs(x) >= 10 else 2)


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def share(x: float) -> str:
    """Anteil (0 bis 1) als ganze Prozent."""
    return f"{de(100.0 * x, 0)} %"


def rho_label(rho: float) -> str:
    """Reaktionsstärke ohne überflüssige Nachkommastelle: 0, 0,5, 1, 2, 4."""
    return de(rho, 1).removesuffix(",0")


def loc_label(c: float) -> str:
    """Halbbreite der Lokalisierung: „aus“ oder die Zahl."""
    return "aus" if c <= 0.0 else de(c, 2)


def count_text(k: int) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{k:,}".replace(",", ".")


# ----------------------------------------------------------------------------- Instanz

def rel_err(x: np.ndarray, xt: np.ndarray) -> float:
    """Fehler in der 2-Norm relativ zur Signalnorm."""
    return float(np.linalg.norm(x - xt) / np.linalg.norm(xt))


def inverse_model(n: int, w: float, rho: float) -> model.Model:
    """Das Modell, das invertiert wird: K_INV Schritte (Stück 9)."""
    return model.Model(model.heat_matrix(n, w, C.K_INV), C.K_INV, rho)


@lru_cache(maxsize=64)
def instance(n: int, w: float, rho: float, noise: float, data: str, seed: int):
    """(s, x, b, sigma): Gitter, Wahrheit (der Anfangswert), Daten und absolutes Rauschen. `honest`: Daten aus dem feineren Modell mit K_TRUE Schritten, `crime`: aus dem invertierten."""
    s = grid(n)
    x = phantom(s)
    fine = inverse_model(n, w, rho) if data == "crime" else model.Model(model.heat_matrix(n, w, C.K_TRUE), C.K_TRUE, rho)
    clean = fine.forward(x)
    sigma = float(noise) * float(np.sqrt(np.mean(clean ** 2)))
    return s, x, clean + SplitMix64(seed).normal(n) * sigma, sigma


def problem(rho: float, noise: float, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W):
    """(inverses Modell, Wahrheit, Daten, sigma, Prior-Kovarianz, Cholesky-Faktor des Priors) für die Experimente: ehrliche Daten, fester Strom SEED_BASE + 1."""
    s, x, b, sigma = instance(n, w, rho, noise, "honest", C.SEED_BASE + 1)
    pcov = model.prior_cov(s)
    return inverse_model(n, w, rho), x, b, sigma, pcov, np.linalg.cholesky(pcov)


def reference(inv: model.Model, b: np.ndarray, sigma: float, pcov: np.ndarray, lc: np.ndarray):
    """(Referenz-Mittel, Referenz-Kovarianz, Art, MAP, MAP-Iterationen, MAP-Jacobi-Matrizen): exakter Posterior bei rho = 0, sonst die Laplace-Näherung am MAP."""
    x_map, iters, jacs = model.map_estimate(inv, b, sigma, lc)
    if inv.rho == 0.0:
        mean, cov = model.linear_posterior(inv.total_matrix(), pcov, sigma, b)
        return mean, cov, "exakt", x_map, iters, jacs
    return x_map, model.laplace_covariance(inv, x_map, sigma, pcov), "Laplace-Näherung", x_map, iters, jacs


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    rho: float = C.DEFAULT_RHO
    data: str = "honest"
    method: str = C.DEFAULT_METHOD
    j: int = C.DEFAULT_J
    steps: int = C.DEFAULT_STEPS
    loc: float = C.DEFAULT_LOC
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    b: np.ndarray
    sigma: float
    steps_run: int            # tatsächliche Schritte (1 beim Update)
    alpha: float
    mean: np.ndarray          # Ensemblemittel (beim Halt nach dem Diskrepanzprinzip: das am Halt)
    sd: np.ndarray
    members: np.ndarray       # die ersten Mitglieder zur Anzeige
    ref_mean: np.ndarray
    ref_sd: np.ndarray
    ref_kind: str
    x_map: np.ndarray
    map_iters: int
    map_jacs: int
    mean_err_sd: float        # Mittelwertabstand zur Referenz in sd (RMS über die Komponenten)
    sd_ratio: float           # Median Ensemble-sd / Referenz-sd
    sd_ratio_min: float
    sd_ratio_max: float
    err: float                # Fehler des Ensemblemittels gegen die Wahrheit
    err_ref: float
    err_map: float
    errs: list                # Fehler des Ensemblemittels nach 0, 1, ... Schritten
    spreads: list
    resid: list
    stop: object              # Iteration des Halts nach dem Diskrepanzprinzip (nur Iteration) oder None
    evals: int
    finite: bool
    diverged: bool
    prior_spread: float       # mittlere sd des Priors, zum Vergleich mit der Streuung des Ensembles


def analyse(settings: Settings) -> Analysis:
    st = settings
    s, x, b, sigma = instance(st.n, st.w, st.rho, st.noise, st.data, st.seed)
    inv = inverse_model(st.n, st.w, st.rho)
    pcov = model.prior_cov(s)
    lc = np.linalg.cholesky(pcov)
    if st.method == "update":
        steps, alpha, disc = 1, 1.0, None
    elif st.method == "tempering":
        steps, alpha, disc = st.steps, float(st.steps), None
    else:
        steps, alpha, disc = st.steps, 1.0, C.DISC_FACTOR
    res = model.run_eki(inv, b, sigma, lc, st.j, steps, alpha, st.seed + 7, model.taper_matrix(s, st.loc), disc)
    ens = res.ens_stop if st.method == "iteration" else res.ens
    ref_mean, cov, kind, x_map, map_iters, map_jacs = reference(inv, b, sigma, pcov, lc)
    ref_sd = np.sqrt(np.diag(cov))
    dm, ratio, rmin, rmax = model.ensemble_stats(ens, ref_mean, ref_sd)
    mean = ens.mean(axis=0)
    err = rel_err(mean, x)
    return Analysis(
        settings=st, s=s, x=x, b=b, sigma=sigma, steps_run=steps, alpha=alpha, mean=mean, sd=ens.std(axis=0, ddof=1), members=ens[:C.N_MEMBERS_SHOWN], ref_mean=ref_mean, ref_sd=ref_sd,
        ref_kind=kind, x_map=x_map, map_iters=map_iters, map_jacs=map_jacs, mean_err_sd=dm, sd_ratio=ratio, sd_ratio_min=rmin, sd_ratio_max=rmax, err=err, err_ref=rel_err(ref_mean, x),
        err_map=rel_err(x_map, x), errs=[rel_err(mu, x) for mu in res.means], spreads=res.spreads, resid=res.resid, stop=res.stop, evals=res.evals, finite=res.finite,
        diverged=bool((not res.finite) or err > C.DIVERGE_ERR), prior_spread=float(np.sqrt(np.diag(pcov)).mean()),
    )


# ----------------------------------------------------------------------------- Experiment 1: Ensemblegröße (linear)

def size_experiment(reps: int = C.SIZE_REPS) -> dict:
    """Linear (rho = 0), ein Update gegen den exakten Posterior: Teil 1 über die Ensemblegröße J bei drei Rauschstufen, Teil 2 über die Dimension n bei J = 50, n + 1 und 4 n."""
    rows = []
    for noise in C.SIZE_NOISE:
        inv, x, b, sigma, pcov, lc = problem(0.0, noise)
        mean, cov = model.linear_posterior(inv.total_matrix(), pcov, sigma, b)
        sd = np.sqrt(np.diag(cov))
        lam = model.identification_eigenvalues(inv.total_matrix(), pcov, sigma)
        for j in C.SIZE_J:
            dms, ratios, errs = [], [], []
            for r in range(reps):
                res = model.run_eki(inv, b, sigma, lc, j, 1, 1.0, C.SEED_BASE + 1000 + r)
                dm, ratio, _, _ = model.ensemble_stats(res.ens, mean, sd)
                dms.append(dm)
                ratios.append(ratio)
                errs.append(rel_err(res.ens.mean(axis=0), x))
            rows.append({"noise": noise, "j": j, "dm": float(np.median(dms)), "dm_min": float(min(dms)), "dm_max": float(max(dms)), "ratio": float(np.median(ratios)),
                         "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios)), "err": float(np.median(errs)), "err_exact": rel_err(mean, x), "directions": int((lam > 1.0).sum())})
    dims = []
    for n in C.DIM_NS:
        inv, x, b, sigma, pcov, lc = problem(0.0, C.DIM_NOISE, n=n)
        mean, cov = model.linear_posterior(inv.total_matrix(), pcov, sigma, b)
        sd = np.sqrt(np.diag(cov))
        lam = model.identification_eigenvalues(inv.total_matrix(), pcov, sigma)
        for j in (50, n + 1, 4 * n):
            stats = [model.ensemble_stats(model.run_eki(inv, b, sigma, lc, j, 1, 1.0, C.SEED_BASE + 4000 + r).ens, mean, sd) for r in range(reps)]
            dims.append({"n": n, "j": j, "dm": float(np.median([q[0] for q in stats])), "ratio": float(np.median([q[1] for q in stats])), "directions": int((lam > 1.0).sum())})
    return {"rows": rows, "dims": dims, "reps": reps, "n": C.DEFAULT_N}


# ----------------------------------------------------------------------------- Experiment 2: Lokalisierung

def localization_experiment(reps: int = C.LOC_REPS) -> dict:
    """Linear, Rauschen LOC_NOISE: ein Update mit Gaspari-Cohn-Lokalisierung (Halbbreite c) gegen den exakten Posterior, nach J und c."""
    inv, x, b, sigma, pcov, lc = problem(0.0, C.LOC_NOISE)
    mean, cov = model.linear_posterior(inv.total_matrix(), pcov, sigma, b)
    sd = np.sqrt(np.diag(cov))
    s = grid(C.DEFAULT_N)
    rows = []
    for j in C.LOC_J:
        for c in C.LOC_RADII:
            taper = model.taper_matrix(s, c)
            stats = [model.ensemble_stats(model.run_eki(inv, b, sigma, lc, j, 1, 1.0, C.SEED_BASE + 7000 + r, taper).ens, mean, sd) for r in range(reps)]
            ratios = [q[1] for q in stats]
            rows.append({"j": j, "c": c, "dm": float(np.median([q[0] for q in stats])), "ratio": float(np.median(ratios)), "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios))})
    return {"rows": rows, "reps": reps}


# ----------------------------------------------------------------------------- Experiment 3: Schritte (nichtlinear)

def steps_experiment(chains: int = C.MCMC_CHAINS, mcmc_steps: int = C.MCMC_STEPS) -> dict:
    """Teil 1: Ensemble (J = STEPS_J) mit N Schritten und alpha = N gegen MCMC (vorkonditioniert mit der Gauss-Newton-Hesse-Matrix, überdispers gestartet) für vier Reaktionsstärken;
    Teil 2: dasselbe im linearen Fall bei J = 100 und 400 gegen den exakten Posterior."""
    rows = []
    for rho in C.STEPS_RHOS:
        inv, x, b, sigma, pcov, lc = problem(rho, C.STEPS_NOISE)
        n = inv.n
        x_map = model.map_estimate(inv, b, sigma, lc)[0]
        lcov = model.laplace_covariance(inv, x_map, sigma, pcov)
        chol = np.linalg.cholesky(lcov + 1e-14 * np.eye(n))
        x0 = x_map[None, :] + 2.0 * SplitMix64(C.SEED_BASE + 5).normal(chains * n).reshape(chains, n) @ chol.T
        r = model.rwm(inv, b, sigma, np.linalg.inv(pcov), x0, mcmc_steps, C.SEED_BASE + 3, C.RWM_SCALE / math.sqrt(n), chol, np.full(n, 1.0 / n))
        total = r["count"] * chains
        mmean = r["sum1"].sum(axis=0) / total
        msd = np.sqrt(np.maximum(r["sum2"].sum(axis=0) / total - mmean ** 2, 0.0))
        burn = int(C.BURN_FRACTION * mcmc_steps)
        row = {"rho": rho, "acc": float(r["acc"].mean()), "rhat": model.rhat(r["traces"][burn:].T), "err_mcmc": rel_err(mmean, x), "err_map": rel_err(x_map, x),
               "laplace_ratio": float(np.median(np.sqrt(np.diag(lcov)) / msd)), "sd_mean": float(msd.mean()), "by_n": {}}
        for steps in C.STEPS_N:
            stats, errs = [], []
            for k in range(C.STEPS_REPS):
                res = model.run_eki(inv, b, sigma, lc, C.STEPS_J, steps, float(steps), C.SEED_BASE + 3000 + k)
                stats.append(model.ensemble_stats(res.ens, mmean, msd))
                errs.append(rel_err(res.ens.mean(axis=0), x))
            ratios = [q[1] for q in stats]
            row["by_n"][steps] = {"dm": float(np.median([q[0] for q in stats])), "ratio": float(np.median(ratios)), "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios)),
                                  "err": float(np.median(errs))}
        rows.append(row)
    inv, x, b, sigma, pcov, lc = problem(0.0, C.STEPS_NOISE)
    mean, cov = model.linear_posterior(inv.total_matrix(), pcov, sigma, b)
    sd = np.sqrt(np.diag(cov))
    linear = []
    for j in C.TEMPER_J:
        for steps in C.TEMPER_N:
            stats = [model.ensemble_stats(model.run_eki(inv, b, sigma, lc, j, steps, float(steps), C.SEED_BASE + 2000 + k).ens, mean, sd) for k in range(C.TEMPER_REPS)]
            linear.append({"j": j, "n": steps, "dm": float(np.median([q[0] for q in stats])), "ratio": float(np.median([q[1] for q in stats]))})
    return {"rows": rows, "linear": linear, "chains": chains, "steps": mcmc_steps}


# ----------------------------------------------------------------------------- Experiment 4: ableitungsfreier Löser

def solver_experiment() -> dict:
    """alpha = 1 iteriert (Halt nach dem Diskrepanzprinzip, SOLVER_ITERS Iterationen insgesamt), Rauschen SOLVER_NOISE, ehrliche Daten, SOLVER_REPS Wiederholungen je Zelle:
    Fehler, Streuung und Residuum je Iteration (Median), Halt und Fehler am Halt; dazu das MAP aus Levenberg-Marquardt (Jacobi-Matrix) als Vergleich."""
    rows = []
    for rho in C.SOLVER_RHOS:
        inv, x, b, sigma, pcov, lc = problem(rho, C.SOLVER_NOISE)
        x_map, iters, jacs = model.map_estimate(inv, b, sigma, lc)
        for j in C.SOLVER_J:
            errs, spreads, resid, stops, err_stop, final = [], [], [], [], [], []
            for k in range(C.SOLVER_REPS):
                res = model.run_eki(inv, b, sigma, lc, j, C.SOLVER_ITERS, 1.0, C.SEED_BASE + 5000 + k, None, C.DISC_FACTOR)
                pad = C.SOLVER_ITERS + 1 - len(res.means)
                errs.append([rel_err(mu, x) for mu in res.means] + [math.nan] * pad)
                spreads.append(list(res.spreads) + [math.nan] * pad)
                resid.append(list(res.resid) + [math.nan] * pad)
                stops.append(res.stop)
                err_stop.append(rel_err(res.ens_stop.mean(axis=0), x))
                final.append(errs[-1][len(res.means) - 1])
            found = [v for v in stops if v is not None]
            rows.append({"rho": rho, "j": j, "errs": [float(v) for v in np.nanmedian(errs, axis=0)], "spreads": [float(v) for v in np.nanmedian(spreads, axis=0)],
                         "resid": [float(v) for v in np.nanmedian(resid, axis=0)], "stop_found": len(found), "stop": float(np.median(found)) if found else None,
                         "err_stop": float(np.median(err_stop)), "err_final": float(np.median(final)), "diverged": int(sum(v > C.DIVERGE_ERR for v in final)),
                         "evals": int((np.median(found) if found else C.SOLVER_ITERS) * j), "err_map": rel_err(x_map, x), "map_iters": iters, "map_jacs": jacs})
    return {"rows": rows, "reps": C.SOLVER_REPS}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    st = a.settings
    bad = []
    if a.diverged:
        bad.append(f"Das Ensemble ist davongelaufen (Fehler {err_pct(a.err)})")
    elif st.method == "iteration":
        if a.stop is None:
            bad.append(f"Das Diskrepanzprinzip ist nach {a.steps_run} Iterationen nicht erreicht (Residuum das {num_text(a.resid[-1])}-Fache des Rauschniveaus)")
        if a.err > C.SOLVER_FACTOR * a.err_map:
            bad.append(f"Fehler {err_pct(a.err)} gegen {err_pct(a.err_map)} beim MAP")
    else:
        if a.sd_ratio < C.SD_LO:
            bad.append(f"Die Streuung ist zu klein: sd-Verhältnis {de(a.sd_ratio, 2)} (unter {de(C.SD_LO, 2)})")
        elif a.sd_ratio > C.SD_HI:
            bad.append(f"Die Streuung ist zu groß: sd-Verhältnis {de(a.sd_ratio, 2)} (über {de(C.SD_HI, 2)})")
        if a.mean_err_sd > C.MEAN_OK:
            bad.append(f"Das Ensemblemittel weicht um {de(a.mean_err_sd, 2)} sd von der Referenz ab (über {de(C.MEAN_OK, 1)})")
    if st.method == "iteration":
        halt = f"Halt nach {a.stop} Iterationen" if a.stop is not None else "kein Halt"
        lead = (f"{C.METHOD_SHORT[st.method]}, {st.j} Mitglieder, {halt} ({count_text(a.evals if a.stop is None else a.stop * st.j)} Vorwärtsläufe): Fehler {err_pct(a.err)} gegen {err_pct(a.err_map)} "
                f"beim MAP ({a.map_jacs} Jacobi-Matrizen), Streuung {de(100.0 * float(np.mean(a.sd)) / a.prior_spread, 1)} % der des Priors")
    else:
        lead = (f"{C.METHOD_SHORT[st.method]}, {st.j} Mitglieder, {a.steps_run} {'Schritt' if a.steps_run == 1 else 'Schritte'} ({count_text(a.evals)} Vorwärtsläufe): Mittel {de(a.mean_err_sd, 2)} sd von der "
                f"Referenz ({a.ref_kind}) entfernt, sd-Verhältnis {de(a.sd_ratio, 2)} ({de(a.sd_ratio_min, 2)} bis {de(a.sd_ratio_max, 2)}), Fehler {err_pct(a.err)} gegen {err_pct(a.err_ref)} beim Referenzmittel")
    lead += "."
    if bad:
        return "warning", f"{lead} Zu beachten: " + "; ".join(bad) + "."
    if st.method == "iteration":
        return "success", f"{lead} Der Löser hält nach dem Diskrepanzprinzip und liegt nahe am MAP."
    return "success", f"{lead} Mittel und Streuung stimmen mit der Referenz überein."


def verdict_size(res: dict) -> tuple[str, str]:
    rows = {(r["noise"], r["j"]): r for r in res["rows"]}
    js = (10, 25, 50, 100, 400)
    r1 = [rows[(1e-2, j)] for j in js]
    return "warning", (f"Bei Rauschen 1 % liegt das sd-Verhältnis des Updates bei J = " + " / ".join(str(j) for j in js) + " bei " + " / ".join(de(r["ratio"], 2) for r in r1)
                       + f"; der Mittelwertabstand fällt von {de(rows[(1e-2, 400)]['dm'], 2)} sd bei J = 400 auf {de(rows[(1e-2, 1600)]['dm'], 2)} sd bei J = 1.600 und {de(rows[(1e-2, 6400)]['dm'], 2)} sd bei J = 6.400. "
                       f"Von {res['n']} Richtungen bestimmen die Daten {rows[(1e-2, 10)]['directions']}; bei Rauschen 0,1 % sind es {rows[(1e-3, 10)]['directions']}, und J = 50 trifft die Streuung nur zu "
                       f"{de(rows[(1e-3, 50)]['ratio'], 2)}.")


def verdict_localization(res: dict) -> tuple[str, str]:
    rows = {(r["j"], r["c"]): r for r in res["rows"]}
    a, b, c = rows[(10, 0.0)], rows[(10, 0.2)], rows[(10, 0.05)]
    return "info", (f"Bei J = 10 hat das Update ohne Lokalisierung das sd-Verhältnis {de(a['ratio'], 2)} und liegt {de(a['dm'], 2)} sd von der Referenz entfernt; mit der Halbbreite 0,2 sind es "
                    f"{de(b['ratio'], 2)} und {de(b['dm'], 2)} sd. Eine zu kleine Halbbreite macht die Unsicherheit zu groß: bei 0,05 ist das Verhältnis {de(c['ratio'], 2)}.")


def verdict_steps(res: dict) -> tuple[str, str]:
    rows = {r["rho"]: r for r in res["rows"]}
    one, four, sixteen = rows[1.0]["by_n"][1], rows[1.0]["by_n"][4], rows[1.0]["by_n"][16]
    return "warning", (f"Bei ρ = 1 ist das einzelne Update in der nichtlinearen Welt zu breit (sd-Verhältnis {de(one['ratio'], 2)} gegen MCMC, Mittel {de(one['dm'], 2)} sd daneben), vier Schritte treffen "
                       f"({de(four['ratio'], 2)}, {de(four['dm'], 2)} sd), sechzehn unterschätzen die Streuung ({de(sixteen['ratio'], 2)}); bei ρ = 4 sind es {de(rows[4.0]['by_n'][16]['ratio'], 2)}. "
                       f"Im linearen Fall senkt jeder weitere Schritt das Verhältnis ebenfalls.")


def verdict_solver(res: dict) -> tuple[str, str]:
    rows = {(r["rho"], r["j"]): r for r in res["rows"]}
    a, b, c = rows[(1.0, 200)], rows[(1.0, 20)], rows[(2.0, 50)]
    return "warning", (f"Bei ρ = 1 hält das Diskrepanzprinzip mit 200 Mitgliedern im Median nach {de(a['stop'], 1)} Iterationen bei einem Fehler von {err_pct(a['err_stop'])} "
                       f"(MAP: {err_pct(a['err_map'])} mit {a['map_jacs']} Jacobi-Matrizen); mit 20 Mitgliedern steht das Ensemble bei {err_pct(b['err_final'])} still. Bei ρ = 2 läuft das Ensemble mit 50 Mitgliedern "
                       f"in {c['diverged']} von {res['reps']} Wiederholungen davon.")
