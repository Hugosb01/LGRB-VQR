"""
=================================================================
VQR — 7-Qubit Model — POST-CNOT CORRECTION LAYERS
=================================================================
Arquitectura por capa (vs. original):

  ORIGINAL:  Ry(x_i) → Rz(θ₁) Ry(θ₂) → CNOT chain
  NUEVA:     Ry(x_i) → Rz(θ₁) Ry(θ₂) → CNOT chain → Rz(φ₁) Ry(φ₂)

Las rotaciones post-CNOT actúan sobre el estado YA entrelazado,
dando al modelo capacidad de corregir cómo el entrelazamiento
afecta a la predicción final.

  Parámetros por capa: 4 × 7 = 28  (antes: 2 × 7 = 14)
  L=2 → 56 parámetros  (antes: 28)

Función de coste: MSE puro (sin restricciones físicas).
Búsqueda adaptativa de capas con PATIENCE = 1.

Output files:
  vqr7q_layers_summary.csv   vqr7q_cv_detail.csv
  vqr7q_predictions.csv
  fig_train_test_curve.png   fig_rmse_curve.png
  fig_boxplot_layers.png     fig_violin_test.png
  fig_scatter_best.png       fig_residuals_best.png
  fig_heatmap_repeats.png
  fig_scatter_all.png        ← scatter todos los GRBs
=================================================================
"""

# ── Thread-count caps MUST be set before any numpy/scipy import ──
import os
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["NUMEXPR_NUM_THREADS"]   = "1"

import sys
import time
import warnings
from scipy.optimize import minimize as sp_minimize
import multiprocessing
import atexit
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from joblib import Parallel, delayed

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import L_BFGS_B

warnings.filterwarnings('ignore')


# =================================================================
# TEE LOGGER — escribe en terminal Y en .log simultáneamente
# =================================================================
class _TeeLogger:
    def __init__(self, logpath):
        self._term = sys.__stdout__
        self._log  = open(logpath, 'w', buffering=1, encoding='utf-8')
        self.logpath = logpath
    def write(self, msg):
        self._term.write(msg)
        self._log.write(msg)
    def flush(self):
        self._term.flush()
        self._log.flush()
    def fileno(self):
        return self._term.fileno()
    def close(self):
        try:
            self._log.flush(); self._log.close()
        except Exception:
            pass


def log(msg=""):
    print(msg, flush=True)


# =================================================================
# CONFIGURATION
# =================================================================
CSV_FILE     = "LGRBs limpio def.csv"
TARGET_NAME  = "log Redshift"
FEATURES = [
    "log10NH", "log10PeakFlux", "PhotonIndex",
    "log10Ta", "log10Fa", "Gamma", "Alpha"
]
N_QUBITS = 7

X_RANGE      = (0, np.pi)
MAXITER      = 150
N_REPEATS_CV = 10
N_FOLDS_CV   = 10

LAYERS_MIN   = 1
LAYERS_MAX   = 6
PATIENCE     = 1
MIN_DELTA    = 0.005

N_JOBS = -1

CLASSICAL_REF_R    = 0.646
CLASSICAL_REF_RMSE = 1.011

# ── NH Ranking loss ──────────────────────────────────────────────
# C_total = MSE + LAMBDA_NH * C_NH_ranking
#
# Para cada par (i,j): si NH_i > NH_j → z_pred_i >= z_pred_j
# Penaliza inversiones del orden físico NH→z.
# Hipótesis: EfficientSU2 (84 params) tiene suficiente capacidad
# para aprender la física NH→z Y mantener buen ajuste estadístico
# simultáneamente (con 28 params no era posible).
#
# LAMBDA_NH = 0.0 → MSE puro
LAMBDA_NH = 0.05
NH_IDX    = 0    # log10NH es la primera columna de FEATURES

LOG_FILE = "vqr_su2_nh.log"
_tee = _TeeLogger(LOG_FILE)
sys.stdout = _tee
atexit.register(_tee.close)


# =================================================================
# PUBLICATION STYLE
# =================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
})

COLOR_PRIMARY   = '#2E5C8A'
COLOR_SECONDARY = '#C44E52'
COLOR_ACCENT    = '#55A868'
COLOR_NEUTRAL   = '#8C8C8C'
COLOR_HIGHLIGHT = '#DD8452'


# =================================================================
# CIRCUIT
# =================================================================
def build_reuploading_circuit(n_qubits, n_layers):
    """
    EfficientSU2 — arquitectura por capa:

        Ry(x_i)              # data re-uploading
        Rz(t1) Ry(t2)        # bloque entrenable 1
        CNOT cadena  0→n     # entrelazamiento izq→der
        Rz(p1) Ry(p2)        # bloque entrenable 2
        CNOT cadena  n→0     # entrelazamiento der→izq (invertido)
        Rz(s1) Ry(s2)        # bloque entrenable 3

    La doble capa CNOT con conectividad alternada (izq→der / der→izq)
    es exactamente el patron EfficientSU2 de Qiskit.
    El primer CNOT propaga correlaciones hacia adelante,
    el segundo las propaga hacia atras, maximizando la mezcla
    de informacion entre todos los pares de qubits tras 2 capas.

    Params por capa: 6 x n_qubits  (post-CNOT tenia: 4 x n_qubits)
    L=2 -> 84 params  (post-CNOT L=2: 56)
    """
    input_params  = [Parameter(f"x{i}") for i in range(n_qubits)]
    n_weights     = 6 * n_qubits * n_layers   # 3 bloques x 2 rotaciones
    weight_params = ParameterVector("theta", n_weights)

    qc    = QuantumCircuit(n_qubits)
    w_idx = 0
    for layer in range(n_layers):

        # ── Encoding ──────────────────────────────────────────────
        for q in range(n_qubits):
            qc.ry(input_params[q], q)

        # ── Bloque entrenable 1 (pre-CNOT) ────────────────────────
        for q in range(n_qubits):
            qc.rz(weight_params[w_idx],     q)
            qc.ry(weight_params[w_idx + 1], q)
            w_idx += 2

        # ── CNOT cadena izquierda → derecha: (0,1),(1,2),...,(n-2,n-1)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

        # ── Bloque entrenable 2 (entre los dos CNOT) ──────────────
        for q in range(n_qubits):
            qc.rz(weight_params[w_idx],     q)
            qc.ry(weight_params[w_idx + 1], q)
            w_idx += 2

        # ── CNOT cadena derecha → izquierda: (n-1,n-2),...,(1,0)
        # Conectividad invertida: propaga correlaciones en sentido opuesto,
        # completando la estructura EfficientSU2 de Qiskit.
        for q in range(n_qubits - 2, -1, -1):
            qc.cx(q + 1, q)

        # ── Bloque entrenable 3 (post-CNOT doble) ─────────────────
        for q in range(n_qubits):
            qc.rz(weight_params[w_idx],     q)
            qc.ry(weight_params[w_idx + 1], q)
            w_idx += 2

        if layer < n_layers - 1:
            qc.barrier()

    return qc, input_params, weight_params


def build_observable(n_qubits):
    obs_list = []
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - q] = "Z"
        obs_list.append(("".join(label), 1.0 / n_qubits))
    return SparsePauliOp.from_list(obs_list)




# =================================================================
# NH RANKING LOSS
# =================================================================

def nh_ranking_loss_and_grad(preds_sc, log_NH, eps=0.01):
    """
    Pairwise ranking loss vectorizada: NH_i > NH_j → pred_i >= pred_j.

    Violación = max(0, eps - sign(ΔNH) * Δpred)

    Implementación vectorizada con np.triu_indices: sin bucles Python,
    compatible con 64 workers en paralelo.

    Devuelve
    --------
    loss      : float  C_NH (sin λ externo)
    grad_pred : array (N,)  ∂C_NH/∂pred_sc_i (sin λ externo)
    """
    N = len(preds_sc)
    i_idx, j_idx = np.triu_indices(N, k=1)

    dNH = log_NH[i_idx] - log_NH[j_idx]
    dP  = preds_sc[i_idx] - preds_sc[j_idx]

    valid = dNH != 0
    if not np.any(valid):
        return 0.0, np.zeros(N)

    sign   = np.sign(dNH[valid])
    i_v    = i_idx[valid]
    j_v    = j_idx[valid]
    dP_v   = dP[valid]

    raw        = eps - sign * dP_v
    violations = np.maximum(0.0, raw)
    n_pairs    = len(violations)

    loss   = float(np.sum(violations) / n_pairs)
    active = (violations > 0).astype(float)

    grad_pred = np.zeros(N)
    np.add.at(grad_pred, i_v, -sign * active / n_pairs)
    np.add.at(grad_pred, j_v, +sign * active / n_pairs)

    return loss, grad_pred


def nh_violation_rate(preds, log_NH):
    """
    Fracción de pares (i,j) donde NH_i > NH_j pero pred_i < pred_j.
    0.0 = orden perfecto, 0.5 = aleatorio.
    Métrica diagnóstica — no entra en el gradiente.
    """
    N = len(preds)
    i_idx, j_idx = np.triu_indices(N, k=1)
    dNH  = log_NH[i_idx] - log_NH[j_idx]
    dP   = preds[i_idx]  - preds[j_idx]
    valid = dNH != 0
    if not np.any(valid):
        return np.nan
    sign = np.sign(dNH[valid])
    return float((sign * dP[valid] < 0).sum()) / valid.sum()


# =================================================================
# REGRESOR CON NH RANKING — bucle manual scipy L-BFGS-B
# =================================================================

def build_regressor_physics(n_qubits, n_layers,
                             X_tr_sc, y_tr_sc, log_NH_tr):
    """
    C_total(θ) = MSE(θ) + LAMBDA_NH * C_NH_ranking(θ)

    Hipótesis: EfficientSU2 (84 params) tiene suficiente capacidad
    para aprender física NH→z Y mantener r estadístico alto,
    a diferencia del modelo original (28 params) donde había trade-off.
    """
    qc, inp, wgt = build_reuploading_circuit(n_qubits, n_layers)
    obs = build_observable(n_qubits)
    qnn = EstimatorQNN(
        circuit=qc, input_params=inp, weight_params=list(wgt),
        observables=[obs], estimator=StatevectorEstimator(),
    )
    N        = len(y_tr_sc)
    n_params = 6 * n_qubits * n_layers
    rng      = np.random.default_rng(seed=42)
    theta0   = rng.uniform(-np.pi, np.pi, n_params)

    def obj(theta):
        preds = qnn.forward(X_tr_sc, theta).ravel()

        # MSE
        res      = preds - y_tr_sc
        mse_loss = float(np.mean(res**2))

        # NH ranking
        nh_loss, nh_grad = nh_ranking_loss_and_grad(preds, log_NH_tr)
        total_loss = mse_loss + LAMBDA_NH * nh_loss

        # Backward (parameter-shift rule)
        _, wg_raw = qnn.backward(X_tr_sc, theta)
        if wg_raw is None:
            raise RuntimeError("qnn.backward() returned None")
        wg = wg_raw[:, 0, :]              # (N, n_params)

        # Regla de la cadena
        mse_g = (2.0 / N) * (res @ wg)
        nh_g  = (LAMBDA_NH * nh_grad) @ wg
        return float(total_loss), mse_g + nh_g

    result = sp_minimize(obj, theta0, method='L-BFGS-B', jac=True,
                         options={'maxiter': MAXITER,
                                  'ftol': 1e-12, 'gtol': 1e-6})
    opt = result.x

    def predict(X_sc):
        return qnn.forward(X_sc, opt).ravel()
    return predict



def build_regressor(n_qubits, n_layers):
    """Each call creates a fresh estimator — safe in parallel workers."""
    qc, inp, wgt = build_reuploading_circuit(n_qubits, n_layers)
    obs           = build_observable(n_qubits)
    estimator     = StatevectorEstimator()   # CPU only, no GPU
    qnn = EstimatorQNN(
        circuit       = qc,
        input_params  = inp,
        weight_params = list(wgt),
        observables   = [obs],
        estimator     = estimator,
    )
    return NeuralNetworkRegressor(
        neural_network = qnn,
        optimizer      = L_BFGS_B(maxiter=MAXITER),
        loss           = "squared_error",
    )


# =================================================================
# RESCALING UTILITIES
# =================================================================
def to_linear_z(y_log10_z_plus_1):
    return np.power(10.0, y_log10_z_plus_1) - 1.0


# =================================================================
# PER-FOLD HELPERS
# =================================================================
def preprocess_fold(X_tr, X_te, y_tr):
    scaler_X = MinMaxScaler(feature_range=X_RANGE)
    X_tr_sc  = scaler_X.fit_transform(X_tr)
    X_te_sc  = scaler_X.transform(X_te)

    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_tr_sc  = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

    return X_tr_sc, X_te_sc, y_tr_sc, scaler_y


def safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def evaluate_fold(X_tr, X_te, y_tr, y_te, n_layers,
                  log_NH_tr, log_NH_te):
    """
    Si LAMBDA_NH > 0: MSE + NH ranking loss
    Si LAMBDA_NH = 0: MSE puro
    """
    X_tr_sc, X_te_sc, y_tr_sc, scaler_y = preprocess_fold(X_tr, X_te, y_tr)

    if LAMBDA_NH > 0.0:
        predict_fn = build_regressor_physics(
            N_QUBITS, n_layers, X_tr_sc, y_tr_sc, log_NH_tr)
        yp_tr_sc = predict_fn(X_tr_sc)
        yp_te_sc = predict_fn(X_te_sc)
    else:
        reg = build_regressor(N_QUBITS, n_layers)
        reg.fit(X_tr_sc, y_tr_sc)
        yp_tr_sc = reg.predict(X_tr_sc)
        yp_te_sc = reg.predict(X_te_sc)

    yp_tr_log = scaler_y.inverse_transform(yp_tr_sc.reshape(-1,1)).ravel()
    yp_te_log = scaler_y.inverse_transform(yp_te_sc.reshape(-1,1)).ravel()
    y_tr_log  = y_tr
    y_te_log  = y_te

    yp_tr_z = to_linear_z(yp_tr_log)
    yp_te_z = to_linear_z(yp_te_log)
    y_tr_z  = to_linear_z(y_tr_log)
    y_te_z  = to_linear_z(y_te_log)

    # Métricas físicas NH
    nh_viol_tr  = nh_violation_rate(yp_tr_sc, log_NH_tr)
    nh_viol_te  = nh_violation_rate(yp_te_sc, log_NH_te)
    r_nh_tr     = safe_corr(log_NH_tr, yp_tr_sc)
    r_nh_te     = safe_corr(log_NH_te, yp_te_sc)

    return {
        "r_train_log":      safe_corr(y_tr_log, yp_tr_log),
        "r_test_log":       safe_corr(y_te_log, yp_te_log),
        "RMSE_train_log":   np.sqrt(mean_squared_error(y_tr_log, yp_tr_log)),
        "RMSE_test_log":    np.sqrt(mean_squared_error(y_te_log, yp_te_log)),
        "R2_test_log":      r2_score(y_te_log, yp_te_log),
        "r_train_z":        safe_corr(y_tr_z, yp_tr_z),
        "r_test_z":         safe_corr(y_te_z, yp_te_z),
        "RMSE_train_z":     np.sqrt(mean_squared_error(y_tr_z, yp_tr_z)),
        "RMSE_test_z":      np.sqrt(mean_squared_error(y_te_z, yp_te_z)),
        "R2_test_z":        r2_score(y_te_z, yp_te_z),
        # Métricas físicas NH
        "nh_viol_train":    float(nh_viol_tr),
        "nh_viol_test":     float(nh_viol_te),
        "r_nh_pred_train":  float(r_nh_tr),
        "r_nh_pred_test":   float(r_nh_te),
        "y_test_log":       y_te_log,
        "yp_test_log":      yp_te_log,
        "y_test_z":         y_te_z,
        "yp_test_z":        yp_te_z,
    }


# =================================================================
# PARALLEL WORKER
# =================================================================
def run_single_fold(rep, fold_idx, train_idx, test_idx,
                    X_all, y_all, n_lay):
    warnings.filterwarnings('ignore')

    n_params = 6 * N_QUBITS * n_lay
    nan_record = {
        "n_layers": n_lay, "n_params": n_params,
        "repeat": rep + 1, "fold": fold_idx + 1,
        "r_train_log": np.nan, "r_test_log": np.nan,
        "RMSE_train_log": np.nan, "RMSE_test_log": np.nan,
        "R2_test_log": np.nan,
        "r_train_z": np.nan, "r_test_z": np.nan,
        "RMSE_train_z": np.nan, "RMSE_test_z": np.nan,
        "R2_test_z": np.nan,
        "nh_viol_train": np.nan, "nh_viol_test": np.nan,
        "r_nh_pred_train": np.nan, "r_nh_pred_test": np.nan,
    }

    try:
        log_NH_tr = X_all[train_idx, NH_IDX]
        log_NH_te = X_all[test_idx,  NH_IDX]

        res = evaluate_fold(
            X_all[train_idx], X_all[test_idx],
            y_all[train_idx], y_all[test_idx],
            n_lay, log_NH_tr, log_NH_te,
        )
        fold_record = {
            "n_layers": n_lay,
            "n_params": n_params,
            "repeat":   rep + 1,
            "fold":     fold_idx + 1,
            **{k: v for k, v in res.items()
               if not k.startswith(("y_", "yp_"))}
        }
        predictions = [
            {
                "n_layers":   n_lay,
                "repeat":     rep + 1,
                "fold":       fold_idx + 1,
                "y_true_log": float(yt_log),
                "y_pred_log": float(yp_log),
                "y_true_z":   float(yt_z),
                "y_pred_z":   float(yp_z),
            }
            for yt_log, yp_log, yt_z, yp_z in zip(
                res["y_test_log"], res["yp_test_log"],
                res["y_test_z"],   res["yp_test_z"],
            )
        ]
        # Return raw arrays for best-fold tracking (numpy → converted to list
        # for pickle safety, reconstructed in main process)
        return (fold_record, predictions,
                res["y_test_z"].tolist(), res["yp_test_z"].tolist(),
                log_NH_te.tolist())

    except Exception as exc:
        return (nan_record, [], None, str(exc), None)


# =================================================================
# 1. LOAD DATA
# =================================================================
log("=" * 70)
log("  VQR 7-QUBIT — EfficientSU2 (doble CNOT) + MSE")
log("=" * 70)

n_cpus = multiprocessing.cpu_count()
n_jobs_actual = n_cpus if N_JOBS == -1 else min(N_JOBS, n_cpus)
log(f"  Log file:      {LOG_FILE}")
log(f"  Logical CPUs:  {n_cpus}")
log(f"  Parallel jobs: {n_jobs_actual}")
log(f"  Arquitectura:  EfficientSU2 — RzRy → CNOT(→) → RzRy → CNOT(←) → RzRy")
log(f"  Params/capa:   6 × {N_QUBITS} = {6*N_QUBITS}")
log(f"  PATIENCE:      {PATIENCE}")
log(f"  Funcion coste: MSE + {LAMBDA_NH} * C_NH_ranking")
log(f"  Hipotesis: EfficientSU2 (84p) resuelve trade-off que 28p no podia")

df    = pd.read_csv(CSV_FILE)
y_all = df[TARGET_NAME].values
X_all = df[FEATURES].values

log(f"\n  File:        {CSV_FILE}")
log(f"  Samples:     {len(df)}")
log(f"  Target:      {TARGET_NAME}  (= log10(z+1))")
log(f"  Features:    {len(FEATURES)} -> {FEATURES}")
log(f"  Qubits:      {N_QUBITS}")
log(f"  CV scheme:   {N_REPEATS_CV} x {N_FOLDS_CV} = "
    f"{N_REPEATS_CV * N_FOLDS_CV} folds per layer count")
log(f"  Layer range: [{LAYERS_MIN}, {LAYERS_MAX}]  patience={PATIENCE}")
log(f"  Reference:   r = {CLASSICAL_REF_R}, RMSE = {CLASSICAL_REF_RMSE}")
log(f"               (Narendra et al. 2025, no bias correction)")


# =================================================================
# 2. ADAPTIVE LAYER LOOP  (layers sequential, folds parallel)
# =================================================================
log("\n" + "=" * 70)
log("  ADAPTIVE LAYER SEARCH  (parallel folds)")
log("=" * 70)

all_fold_results  = []
all_predictions   = []
layer_summary     = []

best_r_test_z     = -np.inf
best_n_layers     = None
best_fold_record  = None
non_improve_count = 0

total_start = time.time()

for n_lay in range(LAYERS_MIN, LAYERS_MAX + 1):
    n_params = 6 * N_QUBITS * n_lay
    log(f"\n  --- L = {n_lay}  |  params = {n_params} ---")
    layer_start = time.time()

    # ── Build ALL (rep, fold) tasks for this layer ──────────────
    tasks = []
    for rep in range(N_REPEATS_CV):
        kf = KFold(n_splits=N_FOLDS_CV, shuffle=True,
                   random_state=rep * 7 + 13)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all)):
            tasks.append((rep, fold_idx,
                          train_idx.copy(), test_idx.copy()))

    total_tasks = len(tasks)
    log(f"    Dispatching {total_tasks} folds to {n_jobs_actual} workers...")

    # ── Run in parallel (loky backend: spawn-safe, no fork issues) ──
    raw_results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(run_single_fold)(rep, fold_idx, train_idx, test_idx,
                                 X_all, y_all, n_lay)
        for rep, fold_idx, train_idx, test_idx in tasks
    )

    # ── Collect results ─────────────────────────────────────────
    fold_results = []
    for (rep, fold_idx, _, _), result_tuple in zip(tasks, raw_results):

        # Unpack — 5 elements on success, 4 on error
        if len(result_tuple) == 5:
            fold_record, preds, y_test_z_list, yp_or_err, log_NH_te_list = result_tuple
        else:
            fold_record, preds, y_test_z_list, yp_or_err = result_tuple
            log_NH_te_list = None

        fold_results.append(fold_record)
        all_fold_results.append(fold_record)
        all_predictions.extend(preds)

        if y_test_z_list is None:
            log(f"      [fold {rep+1}.{fold_idx+1} failed: {yp_or_err}]")
            continue

        yp_test_z_list = yp_or_err
        r_val = fold_record.get("r_test_z", np.nan)
        if not np.isnan(r_val):
            prev_best = (best_fold_record["r_test_z"]
                         if best_fold_record else -np.inf)
            if r_val > prev_best:
                best_fold_record = {
                    **fold_record,
                    "y_test_z":    np.array(y_test_z_list),
                    "yp_test_z":   np.array(yp_test_z_list),
                    "log_NH_test": (np.array(log_NH_te_list)
                                    if log_NH_te_list else None),
                }

    # ── Per-rep progress summary ────────────────────────────────
    for rep in range(N_REPEATS_CV):
        rep_subset = [
            f for f in fold_results
            if f["repeat"] == rep + 1
            and not np.isnan(f.get("r_test_z", np.nan))
        ]
        if rep_subset:
            r_z_mean    = np.mean([f["r_test_z"]       for f in rep_subset])
            r_nh_mean   = np.nanmean([f.get("r_nh_pred_test", np.nan) for f in rep_subset])
            viol_mean   = np.nanmean([f.get("nh_viol_test",   np.nan) for f in rep_subset])
            log(f"      Rep {rep+1:2d}/{N_REPEATS_CV} | "
                f"r_test(z)={r_z_mean:+.4f} | "
                f"r(NH,zpred)={r_nh_mean:+.4f} | "
                f"nh_viol={viol_mean:.4f}")

    # ── Layer summary statistics ────────────────────────────────
    valid = [f for f in fold_results
             if not np.isnan(f.get("r_test_z", np.nan))]

    def m(key): return np.mean([f[key] for f in valid])
    def s(key): return np.std( [f[key] for f in valid])

    layer_time = time.time() - layer_start

    summary = {
        "n_layers": n_lay,
        "n_params": n_params,
        "r_train_log_mean":   round(m("r_train_log"),   4),
        "r_train_log_std":    round(s("r_train_log"),   4),
        "r_test_log_mean":    round(m("r_test_log"),    4),
        "r_test_log_std":     round(s("r_test_log"),    4),
        "RMSE_test_log_mean": round(m("RMSE_test_log"), 4),
        "RMSE_test_log_std":  round(s("RMSE_test_log"), 4),
        "R2_test_log_mean":   round(m("R2_test_log"),   4),
        "r_train_z_mean":     round(m("r_train_z"),     4),
        "r_train_z_std":      round(s("r_train_z"),     4),
        "r_test_z_mean":      round(m("r_test_z"),      4),
        "r_test_z_std":       round(s("r_test_z"),      4),
        "RMSE_test_z_mean":   round(m("RMSE_test_z"),   4),
        "RMSE_test_z_std":    round(s("RMSE_test_z"),   4),
        "R2_test_z_mean":     round(m("R2_test_z"),     4),
        "gap_z":              round(m("r_train_z") - m("r_test_z"), 4),
        "nh_viol_test_mean":  round(m("nh_viol_test"),    4),
        "r_nh_pred_test_mean":round(m("r_nh_pred_test"), 4),
        "n_evaluations":      len(valid),
        "time_minutes":       round(layer_time / 60, 2),
    }
    layer_summary.append(summary)

    log(f"\n   SUMMARY L={n_lay}:")
    log(f"      [log scale]    r_test = {summary['r_test_log_mean']:+.4f} "
        f"± {summary['r_test_log_std']:.4f}")
    log(f"      [linear z]     r_test = {summary['r_test_z_mean']:+.4f} "
        f"± {summary['r_test_z_std']:.4f}  "
        f"(Narendra ref: {CLASSICAL_REF_R})")
    log(f"      [linear z]   RMSE_test = {summary['RMSE_test_z_mean']:.4f} "
        f"± {summary['RMSE_test_z_std']:.4f}  "
        f"(Narendra ref: {CLASSICAL_REF_RMSE})")
    log(f"      gap (linear z) = {summary['gap_z']:+.4f}")
    log(f"      [fisica NH]  nh_viol={summary['nh_viol_test_mean']:.4f}  "
        f"r(NH,zpred)={summary['r_nh_pred_test_mean']:+.4f}")
    log(f"      time = {layer_time/60:.1f} min  "
        f"(wall-clock with {n_jobs_actual} workers)")

    # Incremental save — crash-safe
    pd.DataFrame(layer_summary).to_csv("vqr7q_layers_summary.csv",  index=False)
    pd.DataFrame(all_fold_results).to_csv("vqr7q_cv_detail.csv",    index=False)
    pd.DataFrame(all_predictions).to_csv("vqr7q_predictions.csv",   index=False)

    # ── Adaptive early stopping (linear-z r_test) ───────────────
    if summary["r_test_z_mean"] > best_r_test_z + MIN_DELTA:
        best_r_test_z     = summary["r_test_z_mean"]
        best_n_layers     = n_lay
        non_improve_count = 0
        log(f"      *** NEW BEST (linear z) r = {best_r_test_z:.4f} ***")
    else:
        non_improve_count += 1
        log(f"      no improvement ({non_improve_count}/{PATIENCE})")
        if non_improve_count >= PATIENCE:
            log(f"\n  Early stopping at L={n_lay}.  "
                f"Best L={best_n_layers}  "
                f"(linear-z r={best_r_test_z:.4f})")
            break

total_time = time.time() - total_start


# =================================================================
# 3. FINAL TABLE
# =================================================================
df_layers = pd.DataFrame(layer_summary)

log("\n" + "=" * 70)
log("  FINAL RESULTS  (linear z scale, comparable with Narendra)")
log("=" * 70)
log(f"  {'L':>3s} {'params':>7s} {'r_train':>14s} {'r_test':>14s} "
    f"{'RMSE_test':>14s} {'R2_test':>10s}")
log("  " + "-" * 70)
for _, row in df_layers.iterrows():
    mark = " *" if row["n_layers"] == best_n_layers else "  "
    log(f"  {int(row['n_layers']):>3d} {int(row['n_params']):>7d} "
        f"{row['r_train_z_mean']:.3f}±{row['r_train_z_std']:.3f}  "
        f"{row['r_test_z_mean']:.3f}±{row['r_test_z_std']:.3f}  "
        f"{row['RMSE_test_z_mean']:.3f}±{row['RMSE_test_z_std']:.3f}  "
        f"{row['R2_test_z_mean']:>+8.3f}{mark}")
log("  " + "-" * 70)
log(f"  Narendra et al. (2025), no bias corr:  "
    f"r={CLASSICAL_REF_R}  RMSE={CLASSICAL_REF_RMSE}")
log(f"\n  Best L = {best_n_layers}")
log(f"  Total runtime: {total_time/3600:.2f} hours")


# =================================================================
# 4. PLOTS  (all in linear z, English)
# =================================================================
log("\n" + "=" * 70)
log("  GENERATING FIGURES (linear z scale)")
log("=" * 70)

df_detail    = pd.DataFrame(all_fold_results)
layers_tested = df_layers["n_layers"].values


# FIG 1 — train vs test r curve
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(df_layers["n_layers"], df_layers["r_train_z_mean"],
            yerr=df_layers["r_train_z_std"],
            marker='o', markersize=6, lw=1.5, color=COLOR_PRIMARY,
            capsize=3, capthick=0.8, label='Training', zorder=3)
ax.errorbar(df_layers["n_layers"], df_layers["r_test_z_mean"],
            yerr=df_layers["r_test_z_std"],
            marker='s', markersize=6, lw=1.5, color=COLOR_SECONDARY,
            capsize=3, capthick=0.8, label='Test', zorder=3)
ax.axhline(y=CLASSICAL_REF_R, color='black', ls='--', lw=1.0,
           label=f'Classical benchmark (r={CLASSICAL_REF_R})', zorder=2)
ax.axvline(x=best_n_layers, color=COLOR_NEUTRAL, ls=':', lw=1.0,
           alpha=0.7, zorder=1)
ax.set_xlabel('Number of re-uploading layers, $L$')
ax.set_ylabel('Pearson correlation, $r$ (linear $z$)')
ax.set_title('Training vs test correlation')
ax.legend(loc='best')
ax.set_xticks(layers_tested)
ax.minorticks_on()
plt.tight_layout()
plt.savefig('fig_train_test_curve.png')
plt.close()
log("  -> fig_train_test_curve.png")


# FIG 2 — RMSE curve
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(df_layers["n_layers"], df_layers["RMSE_test_z_mean"],
            yerr=df_layers["RMSE_test_z_std"],
            marker='s', markersize=6, lw=1.5, color=COLOR_SECONDARY,
            capsize=3, capthick=0.8, label='Test')
ax.axhline(y=CLASSICAL_REF_RMSE, color='black', ls='--', lw=1.0,
           label=f'Classical benchmark (RMSE={CLASSICAL_REF_RMSE})')
ax.axvline(x=best_n_layers, color=COLOR_NEUTRAL, ls=':', lw=1.0, alpha=0.7)
ax.set_xlabel('Number of re-uploading layers, $L$')
ax.set_ylabel('RMSE (linear $z$)')
ax.set_title('Test RMSE')
ax.set_xticks(layers_tested)
ax.legend(loc='best')
ax.minorticks_on()
plt.tight_layout()
plt.savefig('fig_rmse_curve.png')
plt.close()
log("  -> fig_rmse_curve.png")


# FIG 3 — boxplot
fig, ax = plt.subplots(figsize=(7, 4.5))
data_box = [df_detail[df_detail["n_layers"] == L]["r_test_z"].dropna().values
            for L in layers_tested]
bp = ax.boxplot(data_box, positions=layers_tested, widths=0.55,
                patch_artist=True,
                medianprops=dict(color='black', lw=1.2),
                flierprops=dict(marker='o', markersize=3,
                                markerfacecolor='gray',
                                markeredgecolor='gray', alpha=0.6),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                boxprops=dict(linewidth=0.8))
for patch in bp['boxes']:
    patch.set_facecolor(COLOR_PRIMARY)
    patch.set_alpha(0.55)
best_idx = list(layers_tested).index(best_n_layers)
bp['boxes'][best_idx].set_facecolor(COLOR_HIGHLIGHT)
bp['boxes'][best_idx].set_alpha(0.75)
ax.axhline(y=CLASSICAL_REF_R, color='black', ls='--', lw=1.0,
           label=f'Classical benchmark (r={CLASSICAL_REF_R})')
ax.axhline(y=0, color='gray', ls='-', lw=0.5, alpha=0.4)
ax.set_xlabel('Number of re-uploading layers, $L$')
ax.set_ylabel(r'Test correlation, $r_{\mathrm{test}}$ (linear $z$)')
ax.set_title(f'Distribution of $r_{{\\mathrm{{test}}}}$ '
             f'({N_REPEATS_CV}$\\times${N_FOLDS_CV} CV)')
ax.set_xticks(layers_tested)
ax.legend(loc='best')
plt.tight_layout()
plt.savefig('fig_boxplot_layers.png')
plt.close()
log("  -> fig_boxplot_layers.png")


# FIG 4 — violin
fig, ax = plt.subplots(figsize=(7, 4.5))
parts = ax.violinplot(data_box, positions=layers_tested, widths=0.7,
                      showmeans=True, showmedians=True, showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(COLOR_HIGHLIGHT
                     if layers_tested[i] == best_n_layers
                     else COLOR_PRIMARY)
    pc.set_edgecolor('black'); pc.set_linewidth(0.6); pc.set_alpha(0.6)
parts['cmeans'].set_color('black');  parts['cmeans'].set_linewidth(1.2)
parts['cmedians'].set_color(COLOR_SECONDARY)
parts['cmedians'].set_linewidth(1.2)
parts['cbars'].set_color('black');   parts['cbars'].set_linewidth(0.6)
parts['cmins'].set_color('black');   parts['cmaxes'].set_color('black')
ax.axhline(y=CLASSICAL_REF_R, color='black', ls='--', lw=1.0,
           label=f'Classical benchmark (r={CLASSICAL_REF_R})')
ax.axhline(y=0, color='gray', ls='-', lw=0.5, alpha=0.4)
ax.set_xlabel('Number of re-uploading layers, $L$')
ax.set_ylabel(r'Test correlation, $r_{\mathrm{test}}$ (linear $z$)')
ax.set_title(r'Density of $r_{\mathrm{test}}$ across CV folds')
ax.set_xticks(layers_tested)
ax.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig('fig_violin_test.png')
plt.close()
log("  -> fig_violin_test.png")


# FIG 5 — best fold scatter
if best_fold_record is not None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    yt = best_fold_record["y_test_z"]
    yp = best_fold_record["yp_test_z"]
    ax.scatter(yt, yp, s=45, alpha=0.75,
               facecolor=COLOR_PRIMARY, edgecolor='black',
               linewidth=0.6, zorder=3)
    lims = [min(yt.min(), yp.min()) - 0.3,
            max(yt.max(), yp.max()) + 0.3]
    ax.plot(lims, lims, color='black', ls='--', lw=1.0, label='1:1', zorder=2)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_xlabel('Observed redshift, $z_{\\mathrm{obs}}$')
    ax.set_ylabel('Predicted redshift, $z_{\\mathrm{pred}}$')
    ax.set_title(f"Best fold (L={best_fold_record['n_layers']})")
    txt = (f"$r$ = {best_fold_record['r_test_z']:.3f}\n"
           f"RMSE = {best_fold_record['RMSE_test_z']:.3f}\n"
           f"$N$ = {len(yt)}")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='black',
                      linewidth=0.6, alpha=0.95))
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('fig_scatter_best.png')
    plt.close()
    log("  -> fig_scatter_best.png")


# FIG 6 — residuals
if best_fold_record is not None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    yt  = best_fold_record["y_test_z"]
    yp  = best_fold_record["yp_test_z"]
    res = yt - yp
    axes[0].hist(res, bins=12, color=COLOR_PRIMARY, alpha=0.7,
                 edgecolor='black', linewidth=0.8)
    axes[0].axvline(0, color='black', ls='--', lw=1.0)
    axes[0].axvline(res.mean(), color=COLOR_SECONDARY, lw=1.2,
                    label=f'Mean = {res.mean():+.3f}')
    axes[0].set_xlabel(r'Residual ($z_{\mathrm{obs}} - z_{\mathrm{pred}}$)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Residual distribution ($\\sigma$={res.std():.3f})')
    axes[0].legend(loc='best')
    axes[1].scatter(yp, res, s=40, alpha=0.7,
                    facecolor=COLOR_PRIMARY, edgecolor='black', linewidth=0.5)
    axes[1].axhline(0, color='black', ls='--', lw=1.0)
    axes[1].set_xlabel(r'Predicted $z_{\mathrm{pred}}$')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals vs predicted')
    plt.tight_layout()
    plt.savefig('fig_residuals_best.png')
    plt.close()
    log("  -> fig_residuals_best.png")


# FIG 7 — heatmap
fig, ax = plt.subplots(figsize=(7, 4.5))
heat = np.full((N_REPEATS_CV, len(layers_tested)), np.nan)
for j, L in enumerate(layers_tested):
    for rep in range(N_REPEATS_CV):
        sub  = df_detail[(df_detail["n_layers"] == L)
                         & (df_detail["repeat"] == rep + 1)]
        vals = sub["r_test_z"].dropna().values
        if len(vals) > 0:
            heat[rep, j] = np.mean(vals)
vmin = np.nanpercentile(heat, 5)
vmax = np.nanpercentile(heat, 95)
im = ax.imshow(heat, aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
for rep in range(N_REPEATS_CV):
    for j in range(len(layers_tested)):
        v = heat[rep, j]
        if not np.isnan(v):
            tc = 'white' if (v - vmin) / (vmax - vmin + 1e-9) > 0.6 else 'black'
            ax.text(j, rep, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color=tc)
ax.set_xticks(range(len(layers_tested)))
ax.set_xticklabels([f'L={L}' for L in layers_tested])
ax.set_yticks(range(N_REPEATS_CV))
ax.set_yticklabels([f'Rep {i+1}' for i in range(N_REPEATS_CV)])
ax.set_xlabel('Number of layers')
ax.set_ylabel('Repetition')
ax.set_title(r'Mean $r_{\mathrm{test}}$ (linear $z$) per repetition and layer')
ax.grid(False)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'Mean $r_{\mathrm{test}}$', rotation=270, labelpad=15)
cbar.outline.set_linewidth(0.6)
plt.tight_layout()
plt.savefig('fig_heatmap_repeats.png')
plt.close()
log("  -> fig_heatmap_repeats.png")



# FIG 8 — Scatter completo: media por GRB sobre todos los folds
# Diagnóstico principal: ¿el post-CNOT estira el rango de predicciones?

# FIG: NH vs z_pred (la figura clave para la defensa)
# Si r(NH,zpred) ≈ r(NH,zobs): EfficientSU2 aprendió la física NH→z
# y además mantiene buen r estadístico → hipótesis confirmada
if (best_fold_record is not None
        and best_fold_record.get("log_NH_test") is not None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    nh    = best_fold_record["log_NH_test"]
    zpred = best_fold_record["yp_test_z"]
    zobs  = best_fold_record["y_test_z"]

    r_nh_pred = safe_corr(nh, zpred)
    r_nh_obs  = safe_corr(nh, zobs)

    for ax, z_val, title, r_val, color in [
        (axes[0], zpred,
         r'$N_H$ vs $z_{\rm pred}$  (colour = $z_{\rm obs}$)',
         r_nh_pred, 'viridis'),
        (axes[1], zobs,
         r'$N_H$ vs $z_{\rm obs}$  (referencia física real)',
         r_nh_obs, 'viridis'),
    ]:
        sc = ax.scatter(nh, z_val, c=zobs, cmap=color,
                        s=55, alpha=0.85,
                        edgecolor='black', linewidths=0.4, zorder=3)
        plt.colorbar(sc, ax=ax).set_label(r'$z_{\rm obs}$', fontsize=10)
        ax.set_xlabel(r'$\log_{10}(N_H\;[\mathrm{cm}^{-2}])$')
        ax.set_ylabel(r'$z$')
        ax.set_title(title)
        ax.text(0.05, 0.95,
                f'$r$ = {r_val:.3f}',
                transform=ax.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='black', lw=0.6, alpha=0.95))

    plt.suptitle(
        f'Física NH→z  (EfficientSU2, $\\lambda_{{NH}}={LAMBDA_NH}$)\n'
        f'Si $r(N_H,z_{{\\rm pred}})\\approx r(N_H,z_{{\\rm obs}})$: '
        f'el modelo aprende física real',
        fontsize=10, y=1.03)
    plt.tight_layout()
    plt.savefig('fig_nh_physics.png', dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  -> fig_nh_physics.png  "
        f"r(NH,zpred)={r_nh_pred:.3f}  r(NH,zobs)={r_nh_obs:.3f}")
else:
    log("  -> fig_nh_physics.png  [omitida: best_fold sin log_NH]")


log("  Generando scatter completo (media por GRB)...")
df_preds_all = pd.DataFrame(all_predictions)
if len(df_preds_all) > 0:
    grb_agg = (df_preds_all.groupby('y_true_z')['y_pred_z']
                           .agg(['mean', 'std'])
                           .reset_index())
    grb_agg.columns = ['z_obs', 'z_pred', 'z_std']
    grb_agg['z_std'] = grb_agg['z_std'].fillna(0)

    z_obs_a  = grb_agg['z_obs'].values
    z_pred_a = grb_agg['z_pred'].values
    z_std_a  = grb_agg['z_std'].values

    r_all     = np.corrcoef(z_obs_a, z_pred_a)[0,1] if np.std(z_pred_a) > 0 else np.nan
    rmse_all  = np.sqrt(np.mean((z_obs_a - z_pred_a)**2))
    bias_all  = float(np.mean(z_pred_a - z_obs_a))
    sp_all    = float(np.std(z_pred_a))
    so_all    = float(np.std(z_obs_a))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    LIMS_S = [-0.3, 9.0]
    sc_a = ax.scatter(z_obs_a, z_pred_a,
                      c=z_obs_a, cmap='viridis',
                      vmin=z_obs_a.min(), vmax=z_obs_a.max(),
                      s=55, alpha=0.85, edgecolor='black',
                      linewidths=0.4, zorder=4)
    ax.errorbar(z_obs_a, z_pred_a, yerr=z_std_a,
                fmt='none', ecolor='gray', elinewidth=0.55,
                alpha=0.40, zorder=3)
    cb_a = plt.colorbar(sc_a, ax=ax, fraction=0.046, pad=0.03)
    cb_a.set_label(r'$z_{\rm obs}$', fontsize=11, labelpad=8)
    cb_a.ax.tick_params(labelsize=10)
    ax.plot(LIMS_S, LIMS_S, color='black', ls='--', lw=1.1,
            label='1:1', zorder=5)
    ax.set_xlim(LIMS_S); ax.set_ylim(LIMS_S); ax.set_aspect('equal')
    ax.minorticks_on()
    ax.set_xlabel(r'Observed redshift, $z_{\rm obs}$')
    ax.set_ylabel(r'Predicted redshift, $\langle z_{\rm pred}\rangle$')
    ax.set_title(
        f'VQR EfficientSU2  ($L^*={best_n_layers}$, '
        f'params={6*N_QUBITS*best_n_layers})\n'
        r'MSE + $\lambda_{NH}\cdot C_{NH\,ranking}$,  '
        f'$\\lambda_{{NH}}={LAMBDA_NH}$'
    )
    txt_a = (f"$r$ = {r_all:.3f}\n"
             f"RMSE = {rmse_all:.3f}\n"
             f"bias = {bias_all:+.3f}\n"
             f"$\\sigma_{{\\rm pred}}$ = {sp_all:.3f}\n"
             f"$\\sigma_{{\\rm obs}}$  = {so_all:.3f}\n"
             f"ratio = {sp_all/so_all:.2f}\n"
             f"$N_{{\\rm GRB}}$ = {len(z_obs_a)}")
    ax.text(0.04, 0.97, txt_a, transform=ax.transAxes,
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                      edgecolor='black', linewidth=0.7, alpha=0.97))
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig('fig_scatter_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  -> fig_scatter_all.png")
    log(f"     r={r_all:.3f}  RMSE={rmse_all:.3f}  "
        f"σ_pred={sp_all:.3f}  σ_obs={so_all:.3f}  "
        f"ratio={sp_all/so_all:.2f}")
else:
    log("  -> fig_scatter_all.png  [omitida: sin predicciones]")


log("\n" + "=" * 70)
log("  DONE")
log("=" * 70)
log(f"  Total runtime:  {total_time/3600:.2f} hours")
log(f"  Workers used:   {n_jobs_actual}")
log(f"  Best L:         {best_n_layers}  (params={6*N_QUBITS*best_n_layers})")
log(f"  Log guardado:   {LOG_FILE}")
log("=" * 70)

_tee.close()
