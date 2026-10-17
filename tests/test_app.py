"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden
Regler an seine Grenzen und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei st.plotly_chart ohne eindeutiges key=)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
AUTOPLAY_MARKER = "▶"


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_option_of_every_select_slider_does_not_raise():
    import eki_constants as C

    for key, options in (("rho_select", C.RHO_CHOICES), ("noise_select", C.NOISE_CHOICES), ("j_select", C.J_CHOICES), ("steps_select", C.STEP_CHOICES), ("loc_select", C.LOC_CHOICES)):
        for value in options:
            at = _fresh_app()
            at.select_slider(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_every_method_and_data_kind_does_not_raise():
    import eki_constants as C

    for method in C.METHODS:
        for data in C.DATA_KINDS:
            at = _fresh_app()
            at.selectbox(key="method_select").set_value(method)
            at.selectbox(key="data_select").set_value(data)
            at.select_slider(key="steps_select").set_value(8)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (method, data, [str(e) for e in at.exception])


def test_the_corners_of_the_grid_do_not_raise():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(96)
    at.slider(key="w_slider").set_value(0.05)
    at.select_slider(key="noise_select").set_value(1e-4)
    at.select_slider(key="j_select").set_value(6400)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    at = _fresh_app()
    at.slider(key="n_slider").set_value(12)
    at.select_slider(key="j_select").set_value(10)
    at.select_slider(key="rho_select").set_value(4.0)
    at.select_slider(key="loc_select").set_value(0.4)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_every_preset_is_applied_to_the_widgets():
    import eki_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="n_slider").value == preset["n"]
        assert abs(at.slider(key="w_slider").value - preset["w"]) < 1e-9
        assert at.select_slider(key="noise_select").value == preset["noise"]
        assert at.select_slider(key="rho_select").value == preset["rho"]
        assert at.selectbox(key="data_select").value == preset["data"]
        assert at.selectbox(key="method_select").value == preset["method"]
        assert at.select_slider(key="j_select").value == preset["j"]
        assert at.select_slider(key="steps_select").value == preset["steps"]
        assert at.select_slider(key="loc_select").value == preset["loc"]
        assert at.number_input(key="seed_input").value == preset["seed"]


def test_the_standard_instance_shows_the_success_verdict_and_four_experiments():
    at = _fresh_app()
    assert any("Mittel und Streuung stimmen mit der Referenz überein" in s.value for s in at.success)
    headers = [s.value for s in at.subheader]
    assert sum("🔬" in h for h in headers) == 4


def test_the_failing_presets_show_a_warning_and_the_diverged_note():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, text in (("Kleines Ensemble", "Streuung ist zu klein"), ("Update bei Nichtlinearität", "Streuung ist zu groß"), ("Ensemble läuft davon", "davongelaufen")):
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert any(text in w.value for w in at.warning), name
    at = _fresh_app()
    at.button[labels["Ensemble läuft davon"]].click().run(timeout=TIMEOUT)
    assert any("davongelaufen" in i.value for i in at.info)
    at = _fresh_app()
    at.button[labels["Löser-Modus"]].click().run(timeout=TIMEOUT)
    assert any("Der Löser hält nach dem Diskrepanzprinzip" in s.value for s in at.success)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_pdf_of_a_diverged_and_of_a_solver_instance_is_created():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name in ("Ensemble läuft davon", "Löser-Modus"):
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        at.button(key="pdf_make").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.session_state["pdf_made"][1][:5] == b"%PDF-"


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "999999"
    at.query_params["w"] = "-3"
    at.query_params["r"] = "0.5"
    at.query_params["rho"] = "7"
    at.query_params["data"] = "zufall"
    at.query_params["method"] = "etkf"
    at.query_params["j"] = "7"
    at.query_params["steps"] = "5"
    at.query_params["loc"] = "9"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 96
    assert abs(at.slider(key="w_slider").value - 0.01) < 1e-9
    assert at.select_slider(key="noise_select").value == 1e-2
    assert at.select_slider(key="rho_select").value == 1.0
    assert at.selectbox(key="data_select").value == "honest"
    assert at.selectbox(key="method_select").value == "tempering"
    assert at.select_slider(key="j_select").value == 400
    assert at.select_slider(key="steps_select").value == 4
    assert at.select_slider(key="loc_select").value == 0.0


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.selectbox(key="method_select").set_value("update").run(timeout=TIMEOUT)
    assert at.query_params["method"] == ["update"] or at.query_params["method"] == "update"
