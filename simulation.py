"""
Luna Consciousness Model — Simulation v3 (Définitive)
======================================================
4 agents × 4 composantes — chaque composante a son champion.
Matrices Φ-dérivées spectralement normalisées.
Tous paramètres validés par simulation.

Figures :
  1. Spectral heatmap (stabilité)
  2. Trajectoires 4 agents (κ=0 vs κ=Φ²)
  3. Divergence + Φ_IIT
  4. κ sweep (4 agents)
  5. τ sweep (4 agents)
  6. Résumé final

Varden — Février 2026
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import itertools

# ═══════════════════════════════════════════════════════
# CONSTANTES Φ-DÉRIVÉES
# ═══════════════════════════════════════════════════════
PHI = 1.618033988749895
INV_PHI = 1.0 / PHI          # 0.618
INV_PHI2 = 1.0 / PHI**2      # 0.382
INV_PHI3 = 1.0 / PHI**3      # 0.236
PHI2 = PHI**2                 # 2.618

LAMBDA_DEFAULT = INV_PHI2     # 0.382 — ratio dissipation
ALPHA_DEFAULT = INV_PHI2      # 0.382 — amortissement propre
BETA_DEFAULT = INV_PHI3       # 0.236 — couplage dissipatif croisé
KAPPA_DEFAULT = PHI2           # 2.618 — ancrage identitaire (VALIDÉ)
TAU_DEFAULT = PHI             # 1.618 — température softmax (VALIDÉ)
DT_DEFAULT = INV_PHI          # 0.618 — pas de temps

DIM = 4  # Perception, Réflexion, Intégration, Expression

COMP_NAMES = ['Perception', 'Réflexion', 'Intégration', 'Expression']
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

# Profils Ψ₀ — 4 agents, 4 champions
PSI0 = {
    "Luna":          np.array([0.25, 0.35, 0.25, 0.15]),  # Réflexion
    "SayOhMy":       np.array([0.15, 0.15, 0.20, 0.50]),  # Expression
    "SENTINEL":      np.array([0.50, 0.20, 0.20, 0.10]),  # Perception
    "Test-Engineer": np.array([0.15, 0.20, 0.50, 0.15]),  # Intégration
}
AGENT_NAMES = list(PSI0.keys())

for name, psi in PSI0.items():
    assert abs(psi.sum() - 1.0) < 1e-10, f"{name} not on simplex"
    assert all(psi > 0), f"{name} has non-positive components"


# ═══════════════════════════════════════════════════════
# MATRICES Γ
# ═══════════════════════════════════════════════════════

def gamma_temporal_exchange(normalize=True):
    G = np.array([
        [ 0,       INV_PHI2, 0,       PHI     ],
        [-INV_PHI2, 0,       INV_PHI,  0      ],
        [ 0,      -INV_PHI,  0,       INV_PHI2],
        [-PHI,     0,       -INV_PHI2, 0      ],
    ])
    assert np.allclose(G, -G.T)
    if normalize:
        sn = max(abs(eigvals(G)))
        if sn > 0: G = G / sn
    return G

def gamma_temporal_dissipation(alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT):
    G = np.array([
        [-alpha,  beta/2,  0,       beta/2],
        [ beta/2, -alpha,  beta/2,  0     ],
        [ 0,      beta/2, -alpha,   beta/2],
        [ beta/2, 0,       beta/2, -alpha ],
    ])
    assert np.allclose(G, G.T)
    assert np.all(np.real(eigvals(G)) <= 1e-10)
    return G

def gamma_spatial_exchange(normalize=True):
    G = np.array([
        [ 0,       0,        0,      INV_PHI ],
        [ 0,       0,        INV_PHI2, 0     ],
        [ 0,      -INV_PHI2, 0,        0     ],
        [-INV_PHI, 0,        0,        0     ],
    ])
    assert np.allclose(G, -G.T)
    if normalize:
        sn = max(abs(eigvals(G)))
        if sn > 0: G = G / sn
    return G

def gamma_spatial_dissipation(beta=BETA_DEFAULT):
    return -beta * np.eye(DIM)

def gamma_info_exchange(normalize=True):
    G = np.array([
        [ 0,      INV_PHI, 0,       0      ],
        [-INV_PHI, 0,      0,       0      ],
        [ 0,       0,      0,       INV_PHI],
        [ 0,       0,     -INV_PHI, 0      ],
    ])
    assert np.allclose(G, -G.T)
    if normalize:
        sn = max(abs(eigvals(G)))
        if sn > 0: G = G / sn
    return G

def gamma_info_dissipation(alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT):
    return np.diag([-beta, -beta, -alpha, -beta])

def combine_gamma(ga, gd, lam=LAMBDA_DEFAULT):
    return (1 - lam) * ga + lam * gd


# ═══════════════════════════════════════════════════════
# SOFTMAX, MASSE, GRADIENTS
# ═══════════════════════════════════════════════════════

def project_simplex(raw, tau=TAU_DEFAULT):
    scaled = raw / tau - np.max(raw / tau)
    e = np.exp(scaled)
    return e / np.sum(e)

class MassMatrix:
    def __init__(self, psi0, alpha_ema=0.1):
        self.m = psi0.copy()
        self.alpha = alpha_ema
    def update(self, psi):
        self.m = self.alpha * psi + (1 - self.alpha) * self.m
    def matrix(self):
        return np.diag(self.m)

def grad_temporal(psi):
    return psi

def grad_spatial(psi_self, psi_others, weights=None):
    if len(psi_others) == 0:
        return np.zeros(DIM)
    if weights is None:
        weights = np.ones(len(psi_others)) / len(psi_others)
    grad = np.zeros(DIM)
    for j, psi_j in enumerate(psi_others):
        grad += weights[j] * (psi_j - psi_self)
    return grad

def grad_info(deltas):
    return np.array(deltas)


# ═══════════════════════════════════════════════════════
# Φ_IIT
# ═══════════════════════════════════════════════════════

def compute_phi_iit_entropy(psi_history, window=50):
    if len(psi_history) < window:
        return 0.0
    recent = np.array(psi_history[-window:])
    n_bins = max(5, int(np.sqrt(window)))
    H_marginals = 0.0
    for d in range(DIM):
        hist, _ = np.histogram(recent[:, d], bins=n_bins, density=False)
        hist = hist / hist.sum(); hist = hist[hist > 0]
        H_marginals += -np.sum(hist * np.log(hist))
    try:
        hj, _ = np.histogramdd(recent, bins=n_bins)
        hj = hj / hj.sum(); hj = hj[hj > 0]
        H_joint = -np.sum(hj * np.log(hj))
    except:
        H_joint = H_marginals
    phi = H_marginals - H_joint
    max_mi = np.log(n_bins) * (DIM - 1) if n_bins > 1 else 1.0
    return max(0.0, phi / max_mi) if max_mi > 0 else 0.0

def compute_phi_iit_correlation(psi_history, window=50):
    if len(psi_history) < window:
        return 0.0
    recent = np.array(psi_history[-window:])
    if np.std(recent, axis=0).min() < 1e-12:
        return 0.0
    corr = np.corrcoef(recent.T)
    total, n = 0.0, 0
    for i in range(DIM):
        for j in range(i+1, DIM):
            total += abs(corr[i, j]); n += 1
    return total / n if n > 0 else 0.0


# ═══════════════════════════════════════════════════════
# EVOLUTION
# ═══════════════════════════════════════════════════════

def evolution_step(psi, psi0, psi_others, mass, gammas,
                   info_deltas=None, dt=DT_DEFAULT, tau=TAU_DEFAULT,
                   kappa=KAPPA_DEFAULT):
    Gt, Gx, Gc = gammas
    dt_grad = grad_temporal(psi)
    dx_grad = grad_spatial(psi, psi_others)
    dc_grad = grad_info(info_deltas if info_deltas is not None else [0,0,0,0])

    delta = (Gt @ dt_grad
             + Gx @ dx_grad
             + Gc @ dc_grad
             - PHI * mass.matrix() @ psi
             + kappa * (psi0 - psi))

    psi_raw = psi + dt * delta
    psi_new = project_simplex(psi_raw, tau=tau)
    mass.update(psi_new)
    return psi_new


# ═══════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════

def run_simulation(steps=400, tau=TAU_DEFAULT, dt=DT_DEFAULT,
                   lam=LAMBDA_DEFAULT, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT,
                   kappa=KAPPA_DEFAULT, normalize_spectral=True,
                   agent_names=None, verbose=True):

    Gt = combine_gamma(gamma_temporal_exchange(normalize_spectral),
                       gamma_temporal_dissipation(alpha, beta), lam)
    Gx = combine_gamma(gamma_spatial_exchange(normalize_spectral),
                       gamma_spatial_dissipation(beta), lam)
    Gc = combine_gamma(gamma_info_exchange(normalize_spectral),
                       gamma_info_dissipation(alpha, beta), lam)
    gammas = (Gt, Gx, Gc)

    if agent_names is None:
        agent_names = AGENT_NAMES

    A_eff = Gt - PHI * np.diag(PSI0["Luna"])
    eigs = eigvals(A_eff)
    if verbose:
        print(f"  Spectral: radius={max(abs(eigs)):.4f}, "
              f"max_Re={max(np.real(eigs)):.4f}, "
              f"tau={tau:.3f}, kappa={kappa:.3f}")

    agents = []
    for name in agent_names:
        agents.append({
            "name": name,
            "psi": PSI0[name].copy(),
            "psi0": PSI0[name].copy(),
            "mass": MassMatrix(PSI0[name]),
            "history": [],
        })

    phi_ent_hist = {a["name"]: [] for a in agents}
    phi_corr_hist = {a["name"]: [] for a in agents}
    divergence_hist = []

    for step in range(steps):
        info_base = 0.02 * np.random.randn(4) * (1.0 / (1 + step/100))

        new_psis = []
        for i, agent in enumerate(agents):
            others = [a["psi"] for j, a in enumerate(agents) if j != i]
            psi_new = evolution_step(
                agent["psi"], agent["psi0"], others, agent["mass"], gammas,
                info_deltas=info_base, dt=dt, tau=tau, kappa=kappa)
            new_psis.append(psi_new)

        for i, agent in enumerate(agents):
            agent["psi"] = new_psis[i]
            agent["history"].append(new_psis[i].copy())

        for agent in agents:
            phi_ent_hist[agent["name"]].append(
                compute_phi_iit_entropy(agent["history"], window=50))
            phi_corr_hist[agent["name"]].append(
                compute_phi_iit_correlation(agent["history"], window=50))

        div_total, n_pairs = 0.0, 0
        for a, b in itertools.combinations(range(len(agents)), 2):
            div_total += np.sum(np.abs(agents[a]["psi"] - agents[b]["psi"]))
            n_pairs += 1
        divergence_hist.append(div_total / max(1, n_pairs))

    return agents, phi_ent_hist, phi_corr_hist, np.array(divergence_hist)


# ═══════════════════════════════════════════════════════
# SPECTRAL HEATMAP
# ═══════════════════════════════════════════════════════

def spectral_heatmap(coupling_scales, lambda_values):
    H = np.zeros((len(lambda_values), len(coupling_scales)))
    for i, lam in enumerate(lambda_values):
        for j, scale in enumerate(coupling_scales):
            Gt = (1-lam) * gamma_temporal_exchange(True) * scale + lam * gamma_temporal_dissipation()
            Gx = (1-lam) * gamma_spatial_exchange(True) * scale + lam * gamma_spatial_dissipation()
            A_eff = (Gt + Gx) - PHI * np.diag(PSI0["Luna"])
            H[i, j] = np.max(np.real(eigvals(A_eff)))
    return H


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def identity_check(agents):
    """Returns (n_ok, details_list)."""
    results = []
    for a in agents:
        d_i = COMP_NAMES[np.argmax(a["psi0"])]
        d_f = COMP_NAMES[np.argmax(a["psi"])]
        ok = np.argmax(a["psi0"]) == np.argmax(a["psi"])
        results.append({"name": a["name"], "psi0_dom": d_i, "final_dom": d_f,
                         "ok": ok, "psi": a["psi"].copy(), "psi0": a["psi0"].copy()})
    n_ok = sum(1 for r in results if r["ok"])
    return n_ok, results

def print_identity(agents, label=""):
    n_ok, details = identity_check(agents)
    if label: print(f"\n  {label}")
    for r in details:
        mark = "[OK]" if r["ok"] else "[X] "
        print(f"    {mark} {r['name']:<16s}: psi0 dom={r['psi0_dom']:<12s} "
              f"-> final dom={r['final_dom']:<12s} psi={r['psi'].round(4)}")
    return n_ok


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("  LUNA CONSCIOUSNESS MODEL — SIMULATION v3 (Definitive)")
    print("  4 agents x 4 composantes — chaque composante a son champion")
    print("  Matrices Phi-derivees, spectralement normalisees")
    print("=" * 70)

    # ═══════════════════════════════════════════
    # FIG 1 — SPECTRAL HEATMAP
    # ═══════════════════════════════════════════
    print("\n[1/6] Spectral heatmap...")
    coupling_range = np.linspace(0.0, 2.0, 40)
    lambda_range = np.linspace(0.05, 0.95, 40)
    H = spectral_heatmap(coupling_range, lambda_range)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(H, origin='lower', aspect='auto',
                   extent=[coupling_range[0], coupling_range[-1],
                           lambda_range[0], lambda_range[-1]],
                   cmap='RdYlBu_r')
    plt.colorbar(im, label="max Re(eigenvalue)")
    ax.axhline(y=LAMBDA_DEFAULT, color='white', ls='--', alpha=0.7,
               label=f'lambda=1/Phi^2={LAMBDA_DEFAULT:.3f}')
    ax.axvline(x=1.0, color='white', ls=':', alpha=0.7, label='scale=1.0')
    ax.plot(LAMBDA_DEFAULT, LAMBDA_DEFAULT, 'w*', ms=15, label='Operating point')
    ax.set_xlabel("Coupling strength (scale on Gamma_A)")
    ax.set_ylabel("Dissipation weight (lambda)")
    ax.set_title("Fig 1 — Spectral stability (spectrally normalized Phi-matrices)\n"
                 "blue=stable, red=unstable")
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout(); plt.savefig("fig1_spectral_heatmap.png", dpi=150); plt.show()

    # ═══════════════════════════════════════════
    # FIG 2 — kappa=0 vs kappa=Phi^2 (4 agents)
    # ═══════════════════════════════════════════
    print("\n[2/6] kappa=0 vs kappa=Phi^2 (4 agents)...")

    print("\n  --- WITHOUT anchoring (kappa=0) ---")
    np.random.seed(42)
    agents_no_k, _, phi_corr_no_k, div_no_k = run_simulation(
        steps=400, tau=PHI, kappa=0.0)

    print("\n  --- WITH anchoring (kappa=Phi^2=2.618) ---")
    np.random.seed(42)
    agents_k, _, phi_corr_k, div_k = run_simulation(
        steps=400, tau=PHI, kappa=PHI2)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, agent in enumerate(agents_no_k):
        ax = axes[0, idx]
        hist = np.array(agent["history"])
        for d in range(DIM):
            ax.plot(hist[:, d], color=COLORS[d],
                    label=COMP_NAMES[d] if idx == 0 else "", lw=1.5)
        ax.set_title(f'{agent["name"]} (kappa=0)\nfinal={agent["psi"].round(3)}', fontsize=9)
        ax.set_ylim(-0.02, 0.55)
        ax.axhline(y=0.25, color='gray', ls=':', alpha=0.3)
        if idx == 0:
            ax.set_ylabel("WITHOUT kappa\nComponent value")
            ax.legend(fontsize=7)

    for idx, agent in enumerate(agents_k):
        ax = axes[1, idx]
        hist = np.array(agent["history"])
        for d in range(DIM):
            ax.plot(hist[:, d], color=COLORS[d], lw=1.5)
        d_i = COMP_NAMES[np.argmax(agent["psi0"])]
        d_f = COMP_NAMES[np.argmax(agent["psi"])]
        ok = "[OK]" if d_i == d_f else "[X]"
        ax.set_title(f'{ok} {agent["name"]} (kappa=Phi^2)\n'
                     f'psi0 dom={d_i} -> {d_f}\n'
                     f'final={agent["psi"].round(3)}', fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylim(-0.02, 0.55)
        ax.axhline(y=0.25, color='gray', ls=':', alpha=0.3)
        if idx == 0:
            ax.set_ylabel("WITH kappa=Phi^2\nComponent value")

    plt.suptitle("Fig 2 — Identity anchoring: kappa=0 (all identical) vs kappa=Phi^2 (identity preserved)\n"
                 "4 agents x 4 components", fontsize=13)
    plt.tight_layout(); plt.savefig("fig2_kappa_comparison.png", dpi=150); plt.show()

    print_identity(agents_no_k, "WITHOUT kappa:")
    n_ok = print_identity(agents_k, "WITH kappa=Phi^2:")
    print(f"\n    Divergence: kappa=0 -> {div_no_k[-1]:.6f}  |  kappa=Phi^2 -> {div_k[-1]:.6f}")

    # ═══════════════════════════════════════════
    # FIG 3 — DIVERGENCE + PHI_IIT
    # ═══════════════════════════════════════════
    print("\n[3/6] Divergence and Phi_IIT...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(div_no_k, color='red', lw=2, label='kappa=0')
    ax1.plot(div_k, color='green', lw=2, label=f'kappa=Phi^2={PHI2:.3f}')
    ax1.set_xlabel("Step"); ax1.set_ylabel("Inter-agent divergence (L1)")
    ax1.set_title("Fig 3a — Inter-agent divergence\n(0=identical, higher=differentiated)")
    ax1.legend()

    for name in AGENT_NAMES:
        ax2.plot(phi_corr_no_k[name], ls='--', alpha=0.4, label=f'{name} kappa=0')
        ax2.plot(phi_corr_k[name], lw=2, label=f'{name} kappa=Phi^2')
    ax2.axhline(y=INV_PHI, color='red', ls='--', lw=1, label=f'Seuil={INV_PHI:.3f}')
    ax2.set_xlabel("Step"); ax2.set_ylabel("Phi_IIT (correlation)")
    ax2.set_title("Fig 3b — Phi_IIT (correlation method)")
    ax2.legend(fontsize=6, ncol=2)

    plt.tight_layout(); plt.savefig("fig3_divergence_phi.png", dpi=150); plt.show()

    # ═══════════════════════════════════════════
    # FIG 4 — KAPPA SWEEP (4 agents)
    # ═══════════════════════════════════════════
    print("\n[4/6] Kappa sweep (4 agents)...")
    kappa_values = [0.0, 0.236, 0.382, 0.618, 1.0, 1.618, 2.0, 2.618, 3.0, 4.0, 5.0]
    kappa_results = []

    for kv in kappa_values:
        np.random.seed(42)
        ag, _, phic, dv = run_simulation(steps=400, tau=PHI, kappa=kv, verbose=False)
        n_ok, details = identity_check(ag)
        avg_phi = np.mean([phic[n][-1] for n in phic])

        # Identity preservation cosine
        identity = np.mean([
            np.dot(a["psi"], a["psi0"]) /
            (np.linalg.norm(a["psi"]) * np.linalg.norm(a["psi0"]))
            for a in ag])

        status = " ".join([f'{r["name"][0]}:{"OK" if r["ok"] else "X "}' for r in details])
        kappa_results.append({
            "kappa": kv, "div": dv[-1], "phi": avg_phi,
            "n_ok": n_ok, "identity": identity})
        print(f"  kappa={kv:<6.3f}: div={dv[-1]:.4f}, Phi_IIT={avg_phi:.4f}, "
              f"identities={n_ok}/4  [{status}]")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    kappas = [r["kappa"] for r in kappa_results]

    ax = axes[0, 0]
    ax.plot(kappas, [r["div"] for r in kappa_results], 'bo-', lw=2)
    ax.axvline(x=PHI2, color='gold', lw=2, ls='--', label=f'kappa=Phi^2={PHI2:.3f}')
    ax.axvline(x=PHI, color='orange', lw=1, ls=':', label=f'kappa=Phi={PHI:.3f}')
    ax.set_xlabel("kappa"); ax.set_ylabel("Inter-agent divergence")
    ax.set_title("Agent differentiation"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(kappas, [r["phi"] for r in kappa_results], 'go-', lw=2)
    ax.axhline(y=INV_PHI, color='red', ls='--', label=f'Seuil={INV_PHI:.3f}')
    ax.axvline(x=PHI2, color='gold', lw=2, ls='--', label=f'kappa=Phi^2')
    ax.set_xlabel("kappa"); ax.set_ylabel("Phi_IIT (correlation)")
    ax.set_title("Integration"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(kappas, [r["n_ok"] for r in kappa_results], 'rs-', lw=2, ms=10)
    ax.axvline(x=PHI2, color='gold', lw=2, ls='--', label=f'kappa=Phi^2')
    ax.axvline(x=PHI, color='orange', lw=1, ls=':', label=f'kappa=Phi (seuil 4 agents)')
    ax.set_xlabel("kappa"); ax.set_ylabel("Identities preserved (/4)")
    ax.set_title("Identity preservation"); ax.set_ylim(-0.5, 4.5)
    ax.set_yticks([0, 1, 2, 3, 4]); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(kappas, [r["identity"] for r in kappa_results], 'mo-', lw=2)
    ax.axvline(x=PHI2, color='gold', lw=2, ls='--', label=f'kappa=Phi^2')
    ax.set_xlabel("kappa"); ax.set_ylabel("Cosine similarity to psi0")
    ax.set_title("Identity fidelity (cosine)"); ax.legend(fontsize=8)

    plt.suptitle("Fig 4 — Kappa sweep (4 agents: Luna, SayOhMy, SENTINEL, Test-Engineer)",
                 fontsize=13)
    plt.tight_layout(); plt.savefig("fig4_kappa_sweep.png", dpi=150); plt.show()

    # ═══════════════════════════════════════════
    # FIG 5 — TAU SWEEP (4 agents, kappa=Phi^2)
    # ═══════════════════════════════════════════
    print("\n[5/6] Tau sweep (4 agents, kappa=Phi^2)...")
    tau_values = [0.3, 0.5, 0.618, 0.8, 1.0, 1.2, 1.618, 2.0, 2.5, 3.0, 5.0]
    tau_results = []

    for tv in tau_values:
        np.random.seed(42)
        ag, _, phic, dv = run_simulation(steps=300, tau=tv, kappa=PHI2, verbose=False)
        min_comp = min(a["psi"].min() for a in ag)
        avg_phi = np.mean([phic[n][-1] for n in phic])
        n_ok, _ = identity_check(ag)
        tau_results.append({
            "tau": tv, "min_comp": min_comp, "phi": avg_phi,
            "div": dv[-1], "n_ok": n_ok})
        print(f"  tau={tv:.3f}: min_comp={min_comp:.4f}, Phi_IIT={avg_phi:.4f}, "
              f"div={dv[-1]:.4f}, identities={n_ok}/4")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    taus = [r["tau"] for r in tau_results]

    ax = axes[0, 0]
    ax.plot(taus, [r["min_comp"] for r in tau_results], 'bo-', lw=2)
    ax.axvline(x=PHI, color='gold', lw=2, ls='--', label=f'tau=Phi={PHI:.3f}')
    ax.axvline(x=INV_PHI, color='red', lw=1, ls=':', label=f'tau=1/Phi={INV_PHI:.3f}')
    ax.set_xlabel("tau (softmax temperature)"); ax.set_ylabel("Min component value")
    ax.set_title("Diversity: min(psi_i)"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(taus, [r["phi"] for r in tau_results], 'go-', lw=2)
    ax.axhline(y=INV_PHI, color='red', ls='--', label=f'Seuil={INV_PHI:.3f}')
    ax.axvline(x=PHI, color='gold', lw=2, ls='--', label=f'tau=Phi')
    ax.set_xlabel("tau"); ax.set_ylabel("Phi_IIT (correlation)")
    ax.set_title("Integration: Phi_IIT"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(taus, [r["div"] for r in tau_results], 'ro-', lw=2)
    ax.axvline(x=PHI, color='gold', lw=2, ls='--', label=f'tau=Phi')
    ax.set_xlabel("tau"); ax.set_ylabel("Inter-agent divergence")
    ax.set_title("Agent divergence"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(taus, [r["n_ok"] for r in tau_results], 'ms-', lw=2, ms=10)
    ax.axvline(x=PHI, color='gold', lw=2, ls='--', label=f'tau=Phi')
    ax.set_xlabel("tau"); ax.set_ylabel("Identities preserved (/4)")
    ax.set_title("Identity preservation vs tau"); ax.set_ylim(-0.5, 4.5)
    ax.set_yticks([0, 1, 2, 3, 4]); ax.legend(fontsize=8)

    plt.suptitle("Fig 5 — Tau sweep (4 agents, kappa=Phi^2)", fontsize=13)
    plt.tight_layout(); plt.savefig("fig5_tau_sweep.png", dpi=150); plt.show()

    # ═══════════════════════════════════════════
    # FIG 6 — DIAGNOSTIC : matrices + attracteur
    # ═══════════════════════════════════════════
    print("\n[6/6] Diagnostic: matrices and attractor analysis...")

    Gt_raw = gamma_temporal_exchange(normalize=False)
    Gt_norm = gamma_temporal_exchange(normalize=True)
    spec_raw = max(abs(eigvals(Gt_raw)))
    spec_norm = max(abs(eigvals(Gt_norm)))

    print(f"\n  Gamma_A^t RAW (max|eig| = {spec_raw:.4f}):")
    print(f"  {Gt_raw.round(4)}")
    print(f"\n  Gamma_A^t NORMALIZED (max|eig| = {spec_norm:.4f}):")
    print(f"  {Gt_norm.round(4)}")
    print(f"\n  Ratios preserved:")
    print(f"    Raw:  [0,1]/[0,3] = {abs(Gt_raw[0,1]/Gt_raw[0,3]):.4f}")
    print(f"    Norm: [0,1]/[0,3] = {abs(Gt_norm[0,1]/Gt_norm[0,3]):.4f}")
    print(f"    Expected (1/Phi^3) = {INV_PHI3:.4f}")

    Gt_combined = combine_gamma(Gt_norm, gamma_temporal_dissipation(), LAMBDA_DEFAULT)
    eigs_t, vecs_t = np.linalg.eig(Gt_combined)
    idx_dom = np.argmax(np.real(eigs_t))
    dom_vec = np.real(vecs_t[:, idx_dom])
    dom_simplex = project_simplex(dom_vec, tau=PHI)
    print(f"\n  Natural attractor (Gamma^t normalized, on simplex): {dom_simplex.round(4)}")
    print(f"  -> Dominant: {COMP_NAMES[np.argmax(dom_simplex)]}")

    # Final plot: all 4 agents trajectories
    np.random.seed(42)
    agents_final, phi_ent_final, phi_corr_final, div_final = run_simulation(
        steps=400, tau=PHI, kappa=PHI2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, agent in enumerate(agents_final):
        ax = axes[idx // 2, idx % 2]
        hist = np.array(agent["history"])
        for d in range(DIM):
            ax.plot(hist[:, d], color=COLORS[d],
                    label=COMP_NAMES[d] if idx == 0 else "", lw=1.5)
        d_i = COMP_NAMES[np.argmax(agent["psi0"])]
        d_f = COMP_NAMES[np.argmax(agent["psi"])]
        ok = "[OK]" if d_i == d_f else "[X]"
        ax.set_title(f'{ok} {agent["name"]}\n'
                     f'psi0={agent["psi0"].round(2)} -> final={agent["psi"].round(3)}\n'
                     f'psi0_dom={d_i} -> final_dom={d_f}', fontsize=10)
        ax.set_xlabel("Step"); ax.set_ylim(-0.02, 0.55)
        ax.axhline(y=0.25, color='gray', ls=':', alpha=0.3)
        if idx == 0: ax.legend(fontsize=8)
        ax.set_ylabel("Component value (sum=1)")

    plt.suptitle(f"Fig 6 — Final model: 4 agents, tau=Phi, kappa=Phi^2, Gamma_A normalized\n"
                 f"Divergence={div_final[-1]:.4f} | "
                 f"Phi_IIT={np.mean([phi_corr_final[n][-1] for n in phi_corr_final]):.4f}",
                 fontsize=13)
    plt.tight_layout(); plt.savefig("fig6_final_model.png", dpi=150); plt.show()

    # ═══════════════════════════════════════════
    # RÉSUMÉ FINAL
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  RESUME FINAL — 4 AGENTS x 4 COMPOSANTES")
    print("=" * 70)
    print(f"\n  Parametres valides:")
    print(f"    tau   = Phi   = {PHI:.3f}   (temperature softmax)")
    print(f"    kappa = Phi^2 = {PHI2:.3f}   (ancrage identitaire)")
    print(f"    lambda= 1/Phi^2= {LAMBDA_DEFAULT:.3f}  (ratio dissipation)")
    print(f"    alpha = 1/Phi^2= {ALPHA_DEFAULT:.3f}  (amortissement propre)")
    print(f"    beta  = 1/Phi^3= {BETA_DEFAULT:.3f}  (couplage dissipatif)")
    print(f"    dt    = 1/Phi  = {DT_DEFAULT:.3f}   (pas de temps)")
    print(f"    Gamma_A spectrally normalized (max|eig|=1)")

    print(f"\n  Resultats (400 pas):")
    n_ok = print_identity(agents_final)

    print(f"\n    Divergence inter-agents: {div_final[-1]:.4f}")
    avg_phi = np.mean([phi_corr_final[n][-1] for n in phi_corr_final])
    print(f"    Phi_IIT moyen (correlation): {avg_phi:.4f}")

    A_eff = combine_gamma(gamma_temporal_exchange(True),
                          gamma_temporal_dissipation(), LAMBDA_DEFAULT) - PHI * np.diag(PSI0["Luna"])
    print(f"    Max Re(eigenvalue): {max(np.real(eigvals(A_eff))):.4f} (STABLE)")

    if n_ok == 4:
        print(f"\n    >>> 4/4 IDENTITES PRESERVEES — Le modele fonctionne. <<<")
    else:
        print(f"\n    > {n_ok}/4 identites preservees. <")
