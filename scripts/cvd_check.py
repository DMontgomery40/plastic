"""Pairwise colour distance of the decision palette under simulated colour-vision deficiency (Machado 2009, severity 1.0).
    python scripts/cvd_check.py
Reports the minimum CIE76 delta-E between any two decision colours for normal, protanopia, deuteranopia, tritanopia."""
import itertools, sys
import numpy as np

M = {
    "normal": np.eye(3),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}
def hex2rgb(h): h = h.lstrip('#'); return np.array([int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])
def lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def delin(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
def lab(rgb_lin):
    X = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ rgb_lin
    X = X / np.array([0.95047, 1.0, 1.08883])
    f = np.where(X > 0.008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])
def min_de(palette):
    out = {}
    for name, m in M.items():
        labs = {k: lab(np.clip(m @ lin(hex2rgb(v)), 0, 1)) for k, v in palette.items()}
        pairs = [(np.linalg.norm(labs[a] - labs[b]), a, b) for a, b in itertools.combinations(palette, 2)]
        out[name] = min(pairs)
    return out
CURRENT = {"commit": "#3fd17a", "rollback": "#ff6b6b", "scale": "#f0b429", "project": "#58a6ff", "readonly": "#94a3b4"}
if __name__ == "__main__":
    for label, pal in [("current", CURRENT)] + [(a, dict(zip(CURRENT, b.split(',')))) for a, b in [(x.split('=')[0], x.split('=')[1]) for x in sys.argv[1:]]]:
        print(label)
        for k, (d, a, b) in min_de(pal).items(): print(f"  {k:13s} min dE {d:5.1f}  ({a} vs {b})")
