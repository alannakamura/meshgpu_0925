# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# ===========================
# 1) CONSTANTES E PARÂMETROS
# ===========================

G = 6.67430e-11               # SI
M_sun = 1.98847e30            # kg
R_sun = 6.957e8               # m
M_jup = 1.89813e27            # kg
AU    = 1.495978707e11        # m
DAY   = 86400.0               # s
PI    = np.pi

# Estrela "sol-like"
M_star = 1.0 * M_sun
R_star = 1.0 * R_sun

# Parâmetros "verdadeiros" do Júpiter (para gerar dados sintéticos)
true = dict(
    P_days = 4332.59,          # 11.86 anos
    e      = 0.0489,
    omega_deg = 273.867,       # argumento do periastro (deg) - valor típico
    i_deg  = 90.0,             # alinhar para haver trânsito
    RpRs   = 0.10045,          # ~ R_J / R_Sun
    Mp_Mj  = 1.0,              # ~ 1 M_jup
    gamma  = 0.0,              # offset RV
    T0     = 0.0               # época de trânsito (definimos 0 para simplificar)
)

# ===========================
# 2) GERAR DADOS SINTÉTICOS
# ===========================

# Grade temporal para fotometria (dias)
# (Para um período tão longo, criamos uma janela de trânsito centrada em T0; isso é típico em campanhas contínuas.)
t_tr = np.linspace(-3.0, 3.0, 800)  # 6 dias ao redor do trânsito

# Grade temporal para RV (espalhar amostras ao longo de alguns anos)
t_rv = np.linspace(0.0, 5*365.25, 60)  # 5 anos com 60 pontos

# Ruídos (ajuste como quiser)
sigma_flux = 2e-4   # 200 ppm (fotometria de boa qualidade)
sigma_rv   = 1.0    # m/s

# ===========================
# 3) MODELOS FÍSICOS
# ===========================

def kepler_E(M, e, tol=1e-12, maxiter=50):
    """ Resolve a Eq. de Kepler: M = E - e sin E (para excentricidade e). """
    E = M.copy()
    for _ in range(maxiter):
        f  = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f/fp
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

def rv_model(t_days, P_days, e, omega_deg, Mp_Mj, i_deg, gamma=0.0, Mstar=M_star):
    """ Modelo kepleriano de RV da estrela induzido pelo planeta. """
    P = P_days * DAY
    omega = np.deg2rad(omega_deg)
    i = np.deg2rad(i_deg)

    Mp = Mp_Mj * M_jup
    # Semi-amplitude K (m/s)
    K = ((2*PI*G)/P)**(1/3) * (Mp*np.sin(i)) / ((Mstar+Mp)**(2/3)) * 1.0/np.sqrt(1-e**2)

    # Fase: assumimos T0 em trânsito, o que implica a anomalia verdadeira ~ 90°-omega no trânsito inferior.
    # Para simplicidade, definimos M=2π(t/P) e absorvemos fase no T0 = 0 (coerente com os dados gerados).
    M_anom = 2*PI*(t_days - true["T0"])/P_days
    M_anom = (M_anom + 2*PI) % (2*PI)

    E = kepler_E(M_anom, e)
    # Anomalia verdadeira
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))

    rv = K*(np.cos(nu+omega) + e*np.cos(omega)) + gamma
    return rv

def semi_major_axis_m(P_days, Mstar=M_star, Mplanet=1.0*M_jup):
    """ a em metros, pela 3a de Kepler (massa estrela + planeta). """
    P = P_days*DAY
    mu = G*(Mstar+Mplanet)
    a = (mu*(P/(2*PI))**2)**(1/3)
    return a

def impact_parameter(P_days, e, omega_deg, i_deg, RpRs, Mstar=M_star, Rstar=R_star, Mplanet=1.0*M_jup):
    """ Impact parameter b aproximado no instante de trânsito (conjunção inferior). """
    omega = np.deg2rad(omega_deg)
    i = np.deg2rad(i_deg)
    a = semi_major_axis_m(P_days, Mstar, Mplanet)
    a_Rs = a / Rstar
    # Fator de correção para excentricidade no instante de trânsito:
    # r = a (1 - e^2) / (1 + e sin ω)  (aprox. conjunção inferior)
    fac = (1 - e**2) / (1 + e*np.sin(omega))
    b = a_Rs * np.cos(i) * fac
    return b

def transit_trapezoid(t_days, P_days, e, omega_deg, i_deg, RpRs, T0=0.0, Rstar=R_star, Mstar=M_star, Mplanet=1.0*M_jup):
    """
    Modelo de trânsito "trapezoidal" fisicamente guiado:
    - profundidade = (Rp/Rs)^2
    - tempos de contato aproximados por geometria:
        T14 ≈ (P/π) * arcsin( (R★/a) * sqrt((1+RpRs)^2 - b^2) / sin i_eff )
        T23 idem com (1 - RpRs)
      (i_eff inclui correção por excentricidade via fator 'fac' utilizado em b).
    - fora do trânsito: fluxo ~ 1.0
    """
    P = P_days
    depth = RpRs**2

    # geometria
    omega = np.deg2rad(omega_deg)
    i = np.deg2rad(i_deg)

    a = semi_major_axis_m(P_days, Mstar, Mplanet)
    a_Rs = a / Rstar
    fac = (1 - e**2) / (1 + e*np.sin(omega))

    # Impact parameter e "inclinação efetiva" (para projeção no instante do trânsito)
    b = a_Rs * np.cos(i) * fac
    sin_i_eff = np.sqrt(1 - (np.cos(i)*fac)**2)  # proxy para projetar caminho aparente

    # Evitar domínios inválidos (sem trânsito)
    # Condição de trânsito: b < 1 + RpRs
    if b >= 1 + RpRs or sin_i_eff <= 1e-12:
        return np.ones_like(t_days)

    # Durações (em dias)
    # Cuidado com domínio do arcsin
    def safe_arcsin(x):
        return np.arcsin(np.clip(x, -1.0, 1.0))

    arg14 = (1.0/a_Rs) * np.sqrt((1+RpRs)**2 - b**2) / np.maximum(sin_i_eff, 1e-12)
    arg23 = (1.0/a_Rs) * np.sqrt(np.maximum((1-RpRs)**2 - b**2, 0.0)) / np.maximum(sin_i_eff, 1e-12)

    T14 = (P/PI) * safe_arcsin(np.clip(arg14, 0.0, 1.0))
    T23 = (P/PI) * safe_arcsin(np.clip(arg23, 0.0, 1.0))

    # Trapezoide: contatos em T0 ± T14/2 (1º e 4º) e T0 ± T23/2 (2º e 3º)
    t1, t2 = T0 - T14/2, T0 - T23/2
    t3, t4 = T0 + T23/2, T0 + T14/2

    flux = np.ones_like(t_days)
    for k, t in enumerate(t_days):
        if t <= t1 or t >= t4:
            flux[k] = 1.0
        elif t2 <= t <= t3:
            flux[k] = 1.0 - depth
        elif t1 < t < t2:
            # rampa ingress
            frac = (t - t1) / (t2 - t1 + 1e-12)
            flux[k] = 1.0 - depth * frac
        elif t3 < t < t4:
            # rampa egress
            frac = (t4 - t) / (t4 - t3 + 1e-12)
            flux[k] = 1.0 - depth * frac
    return flux

# "Observações" sintéticas
flux_true = transit_trapezoid(t_tr, **{k: true[k] for k in ["P_days","e","omega_deg","i_deg","RpRs"]}, T0=true["T0"])
rv_true   = rv_model(t_rv, true["P_days"], true["e"], true["omega_deg"], true["Mp_Mj"], true["i_deg"], true["gamma"])

rng = np.random.default_rng(42)
flux_obs = flux_true + rng.normal(0.0, sigma_flux, size=flux_true.size)
rv_obs   = rv_true   + rng.normal(0.0, sigma_rv,   size=rv_true.size)

# ===========================
# 4) PROBLEMA MULTIOBJETIVO
# ===========================

class ExoJupiterProblem(ElementwiseProblem):
    """
    Variáveis de decisão:
        x = [P_days, RpRs, i_deg, e, omega_deg, Mp_Mj]
    Objetivos:
        f1: MSE fotometria (trânsito)
        f2: MSE RV
    Restrições duras (traduzidas em penalizações nas funções objetivo):
        - b < 1 + RpRs (para haver trânsito) -> caso contrário, penaliza f1
        - domínios físicos [bounds]
    """
    def __init__(self):
        xl = np.array([ 3000.0,  0.05,  85.0, 0.0,   0.0,  0.3])   # lower bounds
        xu = np.array([ 6000.0,  0.15,  95.0, 0.6, 360.0,  3.0])   # upper bounds
        super().__init__(n_var=6, n_obj=2, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        P_days, RpRs, i_deg, e, omega_deg, Mp_Mj = x

        # Fotometria
        flux_mod = transit_trapezoid(
            t_tr, P_days, e, omega_deg, i_deg, RpRs, T0=true["T0"],
            Rstar=R_star, Mstar=M_star, Mplanet=Mp_Mj*M_jup
        )
        # Penalização se não houver trânsito (flux_mod == 1 em tudo) ou geometria impossível:
        # Detectar ausência de trânsito: profundidade observada (~delta_flux) vs 0
        if np.allclose(flux_mod, 1.0, atol=1e-5):
            pen_flux = 1e2  # penaliza forte
        else:
            pen_flux = 0.0

        f1 = np.mean((flux_obs - flux_mod)**2) + pen_flux

        # RV
        rv_mod = rv_model(t_rv, P_days, e, omega_deg, Mp_Mj, i_deg, gamma=true["gamma"])
        f2 = np.mean((rv_obs - rv_mod)**2)

        out["F"] = np.array([f1, f2])

# ===========================
# 5) NSGA-II
# ===========================

problem = ExoJupiterProblem()

pop_size = 120
n_gen    = 150

algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
termination = get_termination("n_gen", n_gen)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    save_history=False,
    verbose=True
)

F = res.F
X = res.X

print("\n===== MELHORES SOLUÇÕES NÃO-DOMINADAS (algumas) =====")
for i in np.argsort(F[:,0]+F[:,1])[:10]:
    print(f"f1={F[i,0]:.3e}  f2={F[i,1]:.3e}  |  "
          f"P={X[i,0]:.2f} d, Rp/Rs={X[i,1]:.4f}, i={X[i,2]:.2f}°, e={X[i,3]:.4f}, ω={X[i,4]:.1f}°, Mp={X[i,5]:.3f} Mj")

# ===========================
# 6) VISUALIZAÇÕES
# ===========================

plt.figure(figsize=(6,5))
plt.scatter(F[:,0], F[:,1], s=18, alpha=0.7)
plt.xlabel("f1 = MSE Fotometria (trânsito)")
plt.ylabel("f2 = MSE RV")
plt.title("Frente de Pareto (NSGA-II): Trânsito vs RV")
plt.tight_layout()

# Escolher a solução com menor (f1+f2) só para ilustrar
best = np.argmin(F.sum(axis=1))
xb = X[best]

# Plot comparativo da curva de luz
flux_best = transit_trapezoid(t_tr, xb[0], xb[3], xb[4], xb[2], xb[1], T0=true["T0"],
                              Rstar=R_star, Mstar=M_star, Mplanet=xb[5]*M_jup)

plt.figure(figsize=(7,4))
plt.plot(t_tr, flux_obs, '.', ms=3, label="Fluxo observado (sintético)")
plt.plot(t_tr, flux_best, '-', lw=2, label="Modelo (melhor f1+f2)")
plt.gca().invert_yaxis()  # quedas para baixo (convencional em curvas de trânsito)
plt.xlabel("Tempo (dias, relativo ao trânsito)")
plt.ylabel("Fluxo normalizado")
plt.title("Curva de Trânsito: observado vs modelo")
plt.legend()
plt.tight_layout()

# Plot comparativo da RV
rv_best = rv_model(t_rv, xb[0], xb[3], xb[4], xb[5], xb[2], gamma=true["gamma"])
plt.figure(figsize=(7,4))
plt.errorbar(t_rv, rv_obs, yerr=sigma_rv, fmt='.', ms=4, label="RV observado (sintético)")
plt.plot(t_rv, rv_best, '-', lw=2, label="Modelo (melhor f1+f2)")
plt.xlabel("Tempo (dias)")
plt.ylabel("RV (m/s)")
plt.title("Curva de Velocidade Radial: observado vs modelo")
plt.legend()
plt.tight_layout()

plt.show()
