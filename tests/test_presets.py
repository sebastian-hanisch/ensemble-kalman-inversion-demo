"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen und die dokumentierten Ausgänge."""

import pytest

import eki_constants as C
import eki_evaluation as ev
import eki_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert C.W_MIN <= p["w"] <= C.W_MAX and (p["w"] - C.W_MIN) / C.W_STEP == pytest.approx(round((p["w"] - C.W_MIN) / C.W_STEP)), name
        assert p["noise"] in C.NOISE_CHOICES and p["rho"] in C.RHO_CHOICES and p["data"] in C.DATA_KINDS and p["method"] in C.METHODS, name
        assert p["j"] in C.J_CHOICES and p["steps"] in C.STEP_CHOICES and p["loc"] in C.LOC_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_presets_show_the_documented_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, lin, small, loc, upd, solver, run = (res[k] for k in C.PRESETS)
    assert ev.verdict_instance(std)[0] == "success" and 0.8 <= std.sd_ratio <= 1.25 and std.mean_err_sd < 0.5
    assert ev.verdict_instance(lin)[0] == "success" and lin.ref_kind == "exakt" and lin.sd_ratio == pytest.approx(0.98, abs=0.04) and lin.mean_err_sd == pytest.approx(0.17, abs=0.08)
    assert small.sd_ratio == pytest.approx(0.23, abs=0.08) and small.mean_err_sd > 2.0 and ev.verdict_instance(small)[0] == "warning"
    assert loc.sd_ratio == pytest.approx(0.83, abs=0.12) and 0.5 < loc.mean_err_sd < 2.0 and ev.verdict_instance(loc)[0] == "warning"
    assert upd.sd_ratio > 2.5 and upd.mean_err_sd > 1.5 and ev.verdict_instance(upd)[0] == "warning"
    assert solver.stop is not None and solver.stop <= 10 and solver.err <= C.SOLVER_FACTOR * solver.err_map and ev.verdict_instance(solver)[0] == "success"
    assert run.diverged and run.stop is None and run.err > C.DIVERGE_ERR and ev.verdict_instance(run)[0] == "warning"


def test_experiment_constants_are_consistent():
    assert C.MCMC_CHAINS >= 20 and C.MCMC_STEPS >= 20000 and C.SIZE_REPS >= 30 and C.SIZE_J[0] == 10 and C.SIZE_J[-1] == 6400 and set(C.SIZE_J) <= set(C.J_CHOICES) | {25}
    assert C.LOC_RADII[0] == 0.0 and set(C.LOC_RADII[1:]) <= set(C.LOC_CHOICES) and C.K_TRUE == 8 * C.K_INV and C.DISC_FACTOR == 1.5 and C.SD_LO < 1.0 < C.SD_HI
    assert C.MU_DOWN < 1.0 < C.MU_UP and 0 < C.BURN_FRACTION < 0.5 and C.DEFAULT_J in C.J_CHOICES and C.DEFAULT_STEPS in C.STEP_CHOICES and C.DEFAULT_METHOD in C.METHODS
    assert set(C.STEPS_N) <= set(C.STEP_CHOICES) and C.STEPS_J in C.J_CHOICES and set(C.SOLVER_J) <= set(C.J_CHOICES) | {20}
