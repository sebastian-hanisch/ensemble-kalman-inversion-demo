"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand: eine Instanz rechnet ein Ensemble (J bis 6.400, n höchstens 96), die Laplace-Referenz und ein Levenberg-Marquardt-MAP, unter einer Sekunde;
# die Experimente nutzen feste Ströme.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Wärmeleitung nach K Schritten (Anteil der Länge 1)
NOISE_CHOICES = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)                   # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-2
RHO_CHOICES = (0.0, 0.5, 1.0, 2.0, 4.0)                          # Reaktionsstärke (Stück 9): 0 ist linear
DEFAULT_RHO = 1.0
DATA_KINDS = ("honest", "crime")
DATA_LABELS = {"honest": "Ehrlich (feineres Modell mit 32 Schritten)", "crime": "Aus demselben Modell (inverse crime, 4 Schritte)"}
DATA_SHORT = {"honest": "ehrlich", "crime": "inverse crime"}
METHODS = ("update", "tempering", "iteration")
METHOD_LABELS = {"update": "Ensemble-Kalman-Update (ein Schritt)", "tempering": "Getemperte Inversion (N Schritte mit α = N, Posterior-Näherung)",
                 "iteration": "Iteration (α = 1, Halt nach dem Diskrepanzprinzip, ableitungsfreier Löser)"}
METHOD_SHORT = {"update": "Update", "tempering": "Tempering", "iteration": "Iteration"}
DEFAULT_METHOD = "tempering"
J_CHOICES = (10, 25, 50, 100, 200, 400, 1600, 6400)               # Ensemblegröße
DEFAULT_J = 400
STEP_CHOICES = (1, 2, 4, 8, 16, 30)                              # Schritte N (Tempering) beziehungsweise höchstens so viele Iterationen
DEFAULT_STEPS = 4
LOC_CHOICES = (0.0, 0.05, 0.1, 0.2, 0.4)                         # Halbbreite der Gaspari-Cohn-Lokalisierung (Träger 2c); 0 = aus
DEFAULT_LOC = 0.0
SEED_MAX, DEFAULT_SEED = 999999, 11

# feste Modellannahmen
K_INV = 4                                                        # Schritte (Wärmeleitung, dann Reaktion) im Modell, das invertiert wird (Stück 9)
K_TRUE = 32                                                      # Schritte im feineren Modell, das die ehrlichen Daten erzeugt
PRIOR_TAU, PRIOR_ELL = 0.5, 0.12                                 # exponentieller Prior wie in Stück 8
MU_START, MU_DOWN, MU_UP, MU_MIN = 1e-2, 0.3, 4.0, 1e-12         # Levenberg-Marquardt für das MAP (Whitening-Koordinaten)
LM_TRIES, LM_MAXIT, LM_TOL = 30, 100, 1e-9
DISC_FACTOR = 1.5                                                # Diskrepanzprinzip: Halt, wenn |G(Mittel) - b| <= 1,5 sigma sqrt(m)
DIVERGE_ERR = 3.0                                                # Fehler über dem Dreifachen der Signalnorm (das Prior-Mittel 0 hätte 1): davongelaufen
SHOW_RANGE = (-1.5, 2.5)
Z_BAND = 2.0
N_MEMBERS_SHOWN = 8

# Verdikt-Schwellen
SD_LO, SD_HI = 0.8, 1.25      # sd-Verhältnis Ensemble / Referenz im Median: darunter zu wenig Streuung, darüber zu viel
MEAN_OK = 0.5                 # Mittelwertabstand zur Referenz in sd (RMS über die Komponenten)
SOLVER_FACTOR = 2.0           # Löser-Modus: Fehler höchstens das Doppelte des MAP-Fehlers

# Referenz-Sampler (Zufallsschritt-Metropolis, vorkonditioniert mit der Gauss-Newton-Hesse-Matrix, Stück 8)
MCMC_CHAINS, MCMC_STEPS, BURN_FRACTION, BLOCK = 20, 20000, 0.2, 500
RWM_SCALE = 2.38

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 990000
SIZE_NOISE = (1e-1, 1e-2, 1e-3)                                  # Experiment 1
SIZE_J = (10, 25, 50, 100, 400, 1600, 6400)
SIZE_REPS = 30
DIM_NS, DIM_NOISE = (24, 48, 96), 1e-2
LOC_J = (10, 25, 50, 100)                                        # Experiment 2
LOC_RADII = (0.0, 0.4, 0.2, 0.1, 0.05)
LOC_REPS, LOC_NOISE = 30, 1e-2
STEPS_RHOS, STEPS_J, STEPS_N, STEPS_REPS, STEPS_NOISE = (0.0, 1.0, 2.0, 4.0), 400, (1, 4, 16), 10, 1e-2   # Experiment 3
TEMPER_J, TEMPER_N, TEMPER_REPS = (100, 400), (1, 2, 4, 8, 16), 30
SOLVER_RHOS, SOLVER_J, SOLVER_ITERS, SOLVER_REPS, SOLVER_NOISE = (1.0, 2.0), (20, 50, 200), 30, 10, 1e-3   # Experiment 4

COLOR_TRUTH = "#2F6B65"
COLOR_ENSEMBLE = "#4c78a8"
COLOR_BAND = "rgba(76, 120, 168, 0.18)"
COLOR_REF = "#222222"
COLOR_MAP = "#D68A2E"
COLOR_GREY = "#8a8a8a"
COLOR_J = ("#c0392b", "#D68A2E", "#2F6B65", "#4c78a8", "#8e44ad", "#7f7f7f", "#222222")

_BASE = {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "rho": 1.0, "data": "honest", "method": "tempering", "j": DEFAULT_J, "steps": DEFAULT_STEPS, "loc": 0.0,
         "seed": DEFAULT_SEED}
PRESETS = {
    "Standard": dict(_BASE),
    "Lineares Update (ρ = 0)": dict(_BASE, rho=0.0, method="update", steps=1),
    "Kleines Ensemble": dict(_BASE, rho=0.0, method="update", steps=1, j=25),
    "Kleines Ensemble, lokalisiert": dict(_BASE, rho=0.0, method="update", steps=1, j=10, loc=0.2),
    "Update bei Nichtlinearität": dict(_BASE, rho=1.0, method="update", steps=1),
    "Löser-Modus": dict(_BASE, rho=1.0, method="iteration", steps=30, j=200, noise=1e-3),
    "Ensemble läuft davon": dict(_BASE, rho=2.0, method="iteration", steps=30, j=50, noise=1e-3, seed=1),
}
PRESET_HELP = {
    "Standard": "Mittlere Reaktion (ρ = 1), vier getemperte Schritte, 400 Mitglieder: Mittel und Streuung liegen nahe an der Referenz.",
    "Lineares Update (ρ = 0)": "Im linearen Fall trifft ein einziges Update mit 400 Mitgliedern den exakten Posterior bis auf Stichprobenrauschen.",
    "Kleines Ensemble": "Nur 25 Mitglieder bei 48 Unbekannten: die Streuung bricht auf unter ein Viertel der Referenz zusammen, das Mittel liegt mehrere sd daneben.",
    "Kleines Ensemble, lokalisiert": "Zehn Mitglieder mit Gaspari-Cohn-Lokalisierung (Halbbreite 0,2): die Streuung liegt bei 0,83 der Referenz und das Mittel noch eine sd daneben; ohne Lokalisierung sind es 0,02 und 3,3 sd.",
    "Update bei Nichtlinearität": "Ein einzelnes Update in der nichtlinearen Welt: die Streuung ist ein Mehrfaches der Referenz und das Mittel liegt weit daneben.",
    "Löser-Modus": "α = 1 iteriert, Halt nach dem Diskrepanzprinzip: ein ableitungsfreier Löser, der nach wenigen Iterationen das Rauschniveau erreicht, aber fast ohne Streuung.",
    "Ensemble läuft davon": "Dasselbe mit nur 50 Mitgliedern bei ρ = 2: das Ensemble verliert seine Streuung, das Residuum wächst, und das Mittel läuft in die Irre.",
}
