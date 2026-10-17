"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 9 (symmetrisch, Zeilensumme 1, Eigenwerte 1, 0,5, 0,1 zu den Vektoren (1,1,1), (1,0,-1), (1,-2,1)), Wahrheit x = (1, 2, 1), Daten b = (1,3, 1,4, 1,3)
    (rauschfrei für das lineare Modell), sigma = 0,1, Prior N(0, I). Handrechnung (Stück 7): Posterior-Mittel (1,153465, 1,653465, 1,153465), Kovarianz mit Gamma_11 = 0,105864,
    Gamma_22 = 0,336634, Gamma_13 = 0,067403 und Eigenwerten 1/101, 1/26, 1/2; Identifikations-Eigenwerte bei sigma = 0,2: (25, 6,25, 0,25).
    Handgebautes Ensemble J = 6: +-e1, +-e2, +-e3 (Mittel 0, Stichproben-Kovarianz C_xx = 0,4 I): C_xg = 0,4 A, C_gg = 0,4 A², Verstärkung K = C_xg (C_gg + alpha sigma² I)^-1 mit den
    Eigenwerten 0,4a / (0,4a² + alpha 0,01); für alpha = 1: 0,97561 / 1,81818 / 2,857143, Mittelwert nach dem Update K b = (1,205575, 1,491289, 1,205575); für alpha = 4: 0,909091 /
    1,428571 / 0,909091, K b = (1,181818, 1,272727, 1,181818); mit dem Taper I (nur die Diagonalen): K = diag(1,237113, 1,095890, 1,237113), K b = (1,608247, 1,534247, 1,608247)."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    return a, x, b


@pytest.fixture
def mini_model(mini):
    """Ein Schritt, rho = 0: G(x) = A x."""
    import eki_model as m

    return m.Model(mini[0], 1, 0.0)


@pytest.fixture
def mini_model_nl(mini):
    """Ein Schritt, rho = 1: R(u) = 2u - u²; G(1, 2, 1) = (0,91, 0,84, 0,91), J = diag(R'(Ax)) A mit R'(Ax) = (-0,6, -0,8, -0,6)."""
    import eki_model as m

    return m.Model(mini[0], 1, 1.0)


@pytest.fixture
def mini_ensemble():
    return np.vstack([np.eye(3), -np.eye(3)])
