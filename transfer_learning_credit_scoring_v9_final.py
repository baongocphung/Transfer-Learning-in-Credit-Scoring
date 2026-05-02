"""
================================================================================
 TRANSFER LEARNING CHO CREDIT SCORING
 Source: Lending Club →  Target: Default of Credit Card Clients
================================================================================

Mục tiêu: dùng dữ liệu Lending Club (LC, ~ 2.26M, 145 đặc trưng) để hỗ trợ
dự đoán xác suất default trên Default of Credit Card Clients (DCCC, 30K, 25
đặc trưng). Hai miền có không gian đặc trưng KHÁC CHIỀU (heterogeneous DA).

  PHẦN 1 — BASELINE (chỉ DCCC, không transfer)
    1.1 Load DCCC
    1.2 Preprocess DCCC
        1.2.1 Sanitize EDUCATION/MARRIAGE
        1.2.2 Clip PAY_X về [-2, 8]
        1.2.3 Type cast & xác định feature lists
    1.3 Train/Test split (80/20 stratified) + StandardScaler
    1.4 Định nghĩa pipelines 3 model
        1.4.1 Logistic Regression
        1.4.2 Random Forest
        1.4.3 XGBoost
    1.5 5-fold StratifiedKFold cross-validation
    1.6 Hold-out evaluation (6,000 mẫu)
    1.7 Lưu kết quả baseline → output/01_baseline.txt

  PHẦN 2 — HFA (Heterogeneous Feature Augmentation, Yang 2020 Eq. 6.8)
    2.1 Load Lending Club
    2.2 Tiền xử lý LC (loại post-loan, imputation, encode, subsample 300K)
    2.3 HFA core
        2.3.1 augment_source() / augment_target()
        2.3.2 _init_PQ()  — Gaussian random projection (JL lemma)
        2.3.3 _proj_frobenius_ball()  — chiếu Frobenius
        2.3.4 _grad_PQ()  — subgradient + active set
        2.3.5 train_hfa()  — alternating optimization (BCD)
        2.3.6 _platt_calibrate()  — hiệu chuẩn xác suất
    2.4 5-fold CV trên DCCC (LC làm source)
    2.5 Hold-out evaluation + 3-case ablation
    2.6 Lưu kết quả HFA → output/02_hfa.txt

  PHẦN 3 — DANN (Domain-Adversarial Neural Network, Ganin 2016)
    3.1 Kiến trúc
        3.1.1 GradientReversalFn
        3.1.2 DomainEncoder
        3.1.3 LabelPredictor
        3.1.4 DomainClassifier
        3.1.5 HeteroDANN (assembled model)
    3.2 Lịch trình
        3.2.1 compute_lambda(p) — λ schedule warm-up
        3.2.2 compute_lr(p)     — LR polynomial decay
    3.3 train_dann()  — vòng lặp huấn luyện step-based
    3.4 predict_dann() và evaluate
    3.5 5-fold CV
    3.6 Hold-out evaluation + 3-case ablation
    3.7 Lưu kết quả DANN → output/03_dann.txt
"""

#SETUP — LIBRARIES, CONSTANTS, PATHS

# --- S.1. Cài đặt thư viện cần thiết (nếu chạy trên Kaggle)-----------
# !pip install -q kaggle xgboost imbalanced-learn

# --- S.2. Imports -------------------------------------------------------
import os
import sys
import json
import warnings
import datetime
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# --- S.3. Imports cho tiền xử lý & đánh giá -----------------------------------
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold, StratifiedShuffleSplit,
                                     cross_val_predict)
from sklearn.metrics         import (roc_auc_score, roc_curve,
                                     confusion_matrix, brier_score_loss)
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import LinearSVC
from xgboost                 import XGBClassifier

# --- S.4. Imports cho DANN (PyTorch) ------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# --- S.5. Hằng số toàn cục ----------------------------------------------------
SEED         = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# [FIX GPU-1] Seed đầy đủ cho GPU: manual_seed_all đảm bảo mọi GPU đều seed.
# benchmark=False tắt cuDNN auto-tuner để kết quả reproducible trên GPU.
# Đánh đổi ~5% tốc độ — có thể bật lại benchmark=True nếu không cần reproducibility tuyệt đối.
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False  # tắt auto-tuner để reproducible trên GPU

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SIZE    = 0.20            # 80/20 hold-out
N_FOLDS      = 5               # 5-fold CV
N_LC_KEEP    = 300_000         # LC stratified subsample size

# --- S.6. Đường dẫn dữ liệu Kaggle --------------------------------------------
# Đường dẫn theo dataset thực tế đã attach trên Kaggle.
# (Các path trùng lặp đã được rút về 1; typo ".csvv" đã sửa.)
DCCC_PATHS = [
    "/kaggle/input/datasets/organizations/uciml/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv",
]
LC_PATHS = [
    "/kaggle/input/datasets/adarshsng/lending-club-loan-data-csv/loan.csv",
]

# --- S.7. Thư mục output ------------------------------------------------------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --- S.8. Helper: ghi log ra file VÀ console ---------------------------------
class TeeLogger:
    """Ghi đồng thời ra file + stdout. Mỗi section mở một file riêng."""
    def __init__(self, filepath, mode="w"):
        self.f = open(filepath, mode, encoding="utf-8")
    def write(self, msg=""):
        print(msg)
        self.f.write(str(msg) + "\n")
        self.f.flush()
    def section(self, title, level=1):
        bar = "=" if level == 1 else ("-" if level == 2 else ".")
        self.write("\n" + bar * 76)
        self.write(f"  {title}")
        self.write(bar * 76)
    def close(self):
        self.f.close()

RUN_LOG = TeeLogger(OUTPUT_DIR / "00_run_log.txt")
RUN_LOG.write(f"[{datetime.datetime.now()}] Run started.")
RUN_LOG.write(f"Device: {DEVICE} | Seed: {SEED}")
RUN_LOG.write(f"Output dir: {OUTPUT_DIR.resolve()}")

# --- S.9. Helper: thước đo đánh giá -------------------------------------------
def find_optimal_threshold_ks(y_true, y_prob):
    """KS-optimal threshold = argmax(TPR − FPR) (Youden 1950)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx]), float(tpr[idx] - fpr[idx])

def evaluate(name, y_true, y_prob, threshold=None):
    """Trả về dict các thước đo: AUC, KS, Brier, Sens, Spec, Acc, threshold."""
    if threshold is None:
        threshold, ks_stat = find_optimal_threshold_ks(y_true, y_prob)
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks_stat = float(np.max(tpr - fpr))
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return OrderedDict([
        ("Model"      , name),
        ("AUC"        , round(roc_auc_score(y_true, y_prob), 4)),
        ("KS"         , round(ks_stat, 4)),
        ("Brier"      , round(brier_score_loss(y_true, y_prob), 4)),
        ("Sens"       , round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)),
        ("Spec"       , round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4)),
        ("Acc"        , round((tp + tn) / len(y_true), 4)),
        ("Thresh"     , round(threshold, 3)),
        ("TP/TN/FP/FN", f"{tp}/{tn}/{fp}/{fn}"),
    ])

def fmt_metrics_row(d):
    return (f"AUC={d['AUC']:.4f} | KS={d['KS']:.4f} | Brier={d['Brier']:.4f} "
            f"| Sens={d['Sens']:.4f} | Spec={d['Spec']:.4f} "
            f"| Acc={d['Acc']:.4f} | Thr={d['Thresh']:.3f}")

# --- S.10. Helper: tìm file dữ liệu khả dụng ---------------------------------
def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         PHẦN 1 — BASELINE (DCCC, no transfer)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
LOG1 = TeeLogger(OUTPUT_DIR / "01_baseline.txt")
LOG1.section("PHẦN 1 — BASELINE: DCCC alone (No Transfer Learning)", level=1)

# ──────────────────────────────────────────────────────────────────────────────
# 1.1. Load DCCC từ Kaggle
# ──────────────────────────────────────────────────────────────────────────────
LOG1.section("1.1. Load DCCC", level=2)

dccc_path = first_existing(DCCC_PATHS)
if dccc_path is None:
    raise FileNotFoundError(
        "Không tìm thấy file DCCC. Đảm bảo dataset 'default-of-credit-card-"
        "clients-dataset' đã được attach vào notebook Kaggle.")
df_dccc = pd.read_csv(dccc_path)
LOG1.write(f"Đã load DCCC từ: {dccc_path}")
LOG1.write(f"Shape gốc: {df_dccc.shape}")

# Chuẩn hóa tên cột target
target_col_dccc = "default.payment.next.month"
if target_col_dccc not in df_dccc.columns:
    # Một số phiên bản dataset rename → 'default'
    candidates = [c for c in df_dccc.columns if "default" in c.lower()]
    if candidates:
        target_col_dccc = candidates[0]
df_dccc["default"] = df_dccc[target_col_dccc].astype(int)
LOG1.write(f"Target: '{target_col_dccc}' → đặt thành cột 'default' (0/1).")
LOG1.write(f"Default rate: {df_dccc['default'].mean():.4f}  "
           f"(Yeh 2009 baseline ≈ 0.2212)")

# ──────────────────────────────────────────────────────────────────────────────
# 1.2. Tiền xử lý DCCC
# ──────────────────────────────────────────────────────────────────────────────
LOG1.section("1.2. Tiền xử lý DCCC", level=2)

# 1.2.1 Sanitize EDUCATION/MARRIAGE codes ngoài codebook (Yeh 2009)
LOG1.write("1.2.1. Sanitize EDUCATION (gộp {0,5,6}→4) và MARRIAGE (0→3).")
EDU_MAP   = {0: 4, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}
MARR_MAP  = {0: 3, 1: 1, 2: 2, 3: 3}
df_dccc["EDUCATION"] = df_dccc["EDUCATION"].map(EDU_MAP).fillna(4).astype(int)
df_dccc["MARRIAGE"]  = df_dccc["MARRIAGE"].map(MARR_MAP).fillna(3).astype(int)

# 1.2.2 Clip PAY_0..PAY_6 về [-2, 8] (codebook định nghĩa)
LOG1.write("1.2.2. Clip PAY_X về [-2, 8] theo codebook Yeh 2009.")
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
for c in PAY_COLS:
    if c in df_dccc.columns:
        df_dccc[c] = df_dccc[c].clip(-2, 8).astype(int)

# 1.2.3 Build feature lists
LOG1.write("1.2.3. Build feature lists.")
EXCLUDE = {"ID", target_col_dccc, "default"}
dccc_numeric = (["LIMIT_BAL", "AGE"]
                + [f"BILL_AMT{i}" for i in range(1, 7)]
                + [f"PAY_AMT{i}"  for i in range(1, 7)])
dccc_nominal = ["SEX", "EDUCATION", "MARRIAGE"]
dccc_ordinal = PAY_COLS
features_dccc = [c for c in (dccc_numeric + dccc_nominal + dccc_ordinal)
                 if c in df_dccc.columns and c not in EXCLUDE]
LOG1.write(f"   Numeric ({len(dccc_numeric)}): {dccc_numeric}")
LOG1.write(f"   Nominal ({len(dccc_nominal)}): {dccc_nominal}")
LOG1.write(f"   Ordinal ({len(dccc_ordinal)}): {dccc_ordinal}")
LOG1.write(f"   Total features = {len(features_dccc)}")

X_dccc = df_dccc[features_dccc].values.astype(float)
y_dccc = df_dccc["default"].values.astype(int)
LOG1.write(f"   X_dccc.shape = {X_dccc.shape}, y_dccc.shape = {y_dccc.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 1.3. Train/Test split + StandardScaler (fit chỉ trên train fold)
# ──────────────────────────────────────────────────────────────────────────────
LOG1.section("1.3. Train/Test split (80/20 stratified) + Scaler", level=2)

X_dc_tr, X_dc_te, y_dc_tr, y_dc_te = train_test_split(
    X_dccc, y_dccc,
    test_size=TEST_SIZE, stratify=y_dccc, random_state=SEED)

LOG1.write(f"Train pool : X={X_dc_tr.shape}, default rate={y_dc_tr.mean():.4f}")
LOG1.write(f"Hold-out   : X={X_dc_te.shape}, default rate={y_dc_te.mean():.4f}")
LOG1.write("StandardScaler sẽ được fit-trong-Pipeline (anti-leakage).")

# ──────────────────────────────────────────────────────────────────────────────
# 1.4. Định nghĩa pipelines cho 3 model
# ──────────────────────────────────────────────────────────────────────────────
LOG1.section("1.4. Định nghĩa pipelines", level=2)

# Class imbalance — scale_pos_weight cho XGBoost
neg, pos = np.bincount(y_dc_tr.astype(int))
spw = neg / max(pos, 1)
LOG1.write(f"Class balance train: non-default={neg}, default={pos}, "
           f"scale_pos_weight={spw:.3f}")

# 1.4.1 Logistic Regression
LOG1.write("1.4.1. Pipeline LogReg: Imputer(median) + StandardScaler + LR(C=0.1)")
pipe_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf", LogisticRegression(C=0.1, max_iter=1000,
                               class_weight="balanced",
                               solver="lbfgs", random_state=SEED))])

# 1.4.2 Random Forest
LOG1.write("1.4.2. Pipeline RandomForest: 200 cây, max_depth=8, balanced")
pipe_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=SEED, n_jobs=-1))])

# 1.4.3 XGBoost
LOG1.write("1.4.3. Pipeline XGBoost: 200 cây, max_depth=4, lr=0.05")
pipe_xgb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf", XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="logloss",
        # [FIX GPU-2] device='cuda' để XGBoost chạy trên GPU Kaggle (T4/P100).
        # Fallback 'cpu' để code chạy được cả trên máy local.
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=SEED, use_label_encoder=False))])

baseline_models = OrderedDict([
    ("LogisticRegression", pipe_lr),
    ("RandomForest",       pipe_rf),
    ("XGBoost",            pipe_xgb),
])

# ──────────────────────────────────────────────────────────────────────────────
# 1.5. 5-fold StratifiedKFold cross-validation
# ──────────────────────────────────────────────────────────────────────────────
LOG1.section("1.5. 5-fold Cross-Validation trên train pool", level=2)
LOG_CV1 = TeeLogger(OUTPUT_DIR / "01a_baseline_cv.txt")
LOG_CV1.section("PHẦN 1 — BASELINE — CHI TIẾT 5-FOLD CROSS-VALIDATION", level=1)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
baseline_cv_results = []

for name, mdl in baseline_models.items():
    LOG_CV1.section(f"Model: {name}", level=2)
    y_prob_cv = cross_val_predict(mdl, X_dc_tr, y_dc_tr,
                                  cv=cv, method="predict_proba")[:, 1]
    metrics = evaluate(f"{name} (CV)", y_dc_tr, y_prob_cv)
    baseline_cv_results.append(metrics)
    LOG_CV1.write(fmt_metrics_row(metrics))
    LOG1.write(f"   {name:18s} | {fmt_metrics_row(metrics)}")

LOG_CV1.close()
LOG1.write(f"Chi tiết → {OUTPUT_DIR}/01a_baseline_cv.txt")

# ──────────────────────────────────────────────────────────────────────────────
# 1.6. Hold-out evaluation (6,000 mẫu)
# ──────────────────────────────────────────────────────────────────────────────
LOG1.section("1.6. Hold-out evaluation (6,000 mẫu)", level=2)
LOG_HO1 = TeeLogger(OUTPUT_DIR / "01b_baseline_holdout.txt")
LOG_HO1.section("PHẦN 1 — BASELINE — CHI TIẾT HOLD-OUT", level=1)

baseline_holdout_results = []
for name, mdl in baseline_models.items():
    LOG_HO1.section(f"Model: {name}", level=2)
    mdl.fit(X_dc_tr, y_dc_tr)
    y_prob = mdl.predict_proba(X_dc_te)[:, 1]
    # Threshold KS-optimal được chọn TRÊN TRAIN — apply lên test
    y_prob_tr = mdl.predict_proba(X_dc_tr)[:, 1]
    thr_tr, _ = find_optimal_threshold_ks(y_dc_tr, y_prob_tr)
    metrics = evaluate(f"{name} (HoldOut)", y_dc_te, y_prob, threshold=thr_tr)
    baseline_holdout_results.append(metrics)
    LOG_HO1.write(fmt_metrics_row(metrics))
    LOG_HO1.write(f"   confusion = {metrics['TP/TN/FP/FN']}")
    LOG1.write(f"   {name:18s} | {fmt_metrics_row(metrics)}")

LOG_HO1.close()
LOG1.write(f"Chi tiết → {OUTPUT_DIR}/01b_baseline_holdout.txt")

# ──────────────────────────────────────────────────────────────────────────────
# 1.7. Tóm tắt phần BASELINE
# ──────────────────────────────────────────────────────────────────────────────
LOG1.section("1.7. Tóm tắt phần BASELINE", level=2)
LOG1.write("Bảng tổng hợp Baseline (Hold-out):")
hdr = f"  {'Model':<20s} {'AUC':>8s} {'KS':>8s} {'Brier':>8s} {'Sens':>8s} {'Spec':>8s}"
LOG1.write(hdr)
LOG1.write("  " + "-" * 64)
for r in baseline_holdout_results:
    LOG1.write(f"  {r['Model']:<20s} {r['AUC']:>8.4f} {r['KS']:>8.4f} "
               f"{r['Brier']:>8.4f} {r['Sens']:>8.4f} {r['Spec']:>8.4f}")
LOG1.close()
RUN_LOG.write(f"[Done] PHẦN 1 — BASELINE → {OUTPUT_DIR}/01_baseline.txt")


# ════════════════════════════════════════════════════════════════════════
# PHẦN 2 — HFA (Heterogeneous Feature Augmentation)
# ════════════════════════════════════════════════════════════════════════
#
# NGUỒN LÝ THUYẾT BÁM SÁT:
#   - Yang Q., Zhang Y., Dai W., Pan S.J. (2020) Transfer Learning,
#     Cambridge University Press, Chapter 6, Eq. 6.8 (HFA primal form).
#   - Duan L., Xu D., Tsang I.W. (2012) "Learning with Augmented Features
#     for Heterogeneous Domain Adaptation", ICML.
#   - Boyd S., Vandenberghe L. (2004) Convex Optimization §8.1.1.
#   - Tseng P. (2001) J. Optim. Theory Appl. 109:475-494.
#   - Achlioptas D. (2003) J. Comput. Syst. Sci. 66:671-687.
#   - Chapelle O. (2007) Neural Comput. 19:1155-1178 (squared-hinge).
#   - Hsieh C.-J. et al. (2008) ICML (LinearSVC dual coordinate descent).
#   - Platt J.C. (1999) Adv. Large Margin Classifiers Sec. 5.
#
# PHÁT BIỂU HFA (Yang 2020 §6.3.2):
#   Cho hai miền nguồn S = {(x_s,i, y_s,i)} với x_s ∈ R^{d_s},
#   và đích T = {(x_t,j, y_t,j)} với x_t ∈ R^{d_t}, d_s ≠ d_t.
#   Học hai phép chiếu P, Q và hyperplane (w, b) trên không gian augment
#   R^{d_c+d_s+d_t}:
#
#     phi_s(x) = [P x ; x ; 0_{d_t}]      (cho điểm nguồn)
#     phi_t(x) = [Q x ; 0_{d_s} ; x]      (cho điểm đích)
#
#   bằng cách giải hàm mục tiêu Eq. 6.8:
#
#     min  (1/2)||w||^2 + (lam_P/2)||P||_F^2 + (lam_Q/2)||Q||_F^2 + C * sum xi_i^2
#     s.t. y_i (w . phi(x_i) + b) >= 1 - xi_i,  xi_i >= 0,
#          ||P||_F <= r_P,  ||Q||_F <= r_Q.
#
# CHIẾN LƯỢC TỐI ƯU (BCD - Tseng 2001 Th. 4.1):
#   Lặp n_outer vòng, mỗi vòng:
#     (1) Cố định (P, Q) → augment → fit LinearSVC → cập nhật (w, b).
#     (2) Cố định (w, b) → projected sub-gradient trên (P, Q).
#     (3) Chiếu (P, Q) lên cầu Frobenius (closed-form Boyd 2004).
#   Tseng 2001 đảm bảo: mọi điểm tụ là stationary point của Eq. 6.8.
#
# ════════════════════════════════════════════════════════════════════════
LOG2 = TeeLogger(OUTPUT_DIR / "02_hfa.txt")
LOG2.section("PHẦN 2 — HFA: Heterogeneous Feature Augmentation", level=1)
LOG2.write("Source: Lending Club (pipeline R gốc: drop>90%missing/leakage → impute → binary/label/OHE → subsample 300K)")
LOG2.write("Target: DCCC (đã chuẩn bị ở Phần 1)")
LOG2.write("Tham chiếu: Yang Q. et al. (2020) Transfer Learning, Cambridge UP, "
           "Ch.6 Eq.6.8; Duan L., Xu D., Tsang I.W. (2012) ICML.")

# ──────────────────────────────────────────────────────────────────────────────
# 2.1. Load Lending Club
# ──────────────────────────────────────────────────────────────────────────────
LOG2.section("2.1. Load Lending Club", level=2)

lc_path = first_existing(LC_PATHS)
if lc_path is None:
    raise FileNotFoundError(
        "Không tìm thấy file Lending Club. Attach 'wendykan/lending-club-loan-"
        "data' hoặc 'lending-club' lên Kaggle notebook trước khi chạy.")
LOG2.write(f"Đang đọc LC từ: {lc_path}")

# Đọc toàn bộ file LC (145 cột gốc), sau đó lọc dần như pipeline R.
# Không dùng usecols cứng để giữ linh hoạt — pd sẽ tự skip cột không cần.
df_lc_raw = pd.read_csv(lc_path, low_memory=False)
LOG2.write(f"LC raw shape (toàn bộ cột): {df_lc_raw.shape}")
LOG2.write(f"loan_status counts:\n{df_lc_raw['loan_status'].value_counts().head(10)}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.2. Tiền xử lý LC  (khớp pipeline R gốc)
# ──────────────────────────────────────────────────────────────────────────────
LOG2.section("2.2. Tiền xử lý LC", level=2)

# ── 2.2.1. Drop missing > 90% (hardship / settlement / sec_app / joint) ──────
LOG2.section("2.2.1. Drop missing > 90%", level=3)
DROP_HIGH_MISSING = [
    "id", "member_id", "url",
    "orig_projected_additional_accrued_interest",
    "hardship_amount", "hardship_dpd", "hardship_loan_status",
    "deferral_term", "hardship_end_date", "hardship_status",
    "hardship_start_date", "hardship_reason", "hardship_type",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "hardship_length", "payment_plan_start_date",
    "debt_settlement_flag_date", "settlement_term", "settlement_amount",
    "settlement_date", "settlement_percentage", "settlement_status",
    "sec_app_mths_since_last_major_derog", "sec_app_revol_util",
    "revol_bal_joint", "sec_app_earliest_cr_line", "sec_app_mort_acc",
    "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_inq_last_6mths", "sec_app_chargeoff_within_12_mths",
    "sec_app_open_acc", "sec_app_collections_12_mths_ex_med",
    "verification_status_joint", "dti_joint", "annual_inc_joint", "desc",
]
drop_existing = [c for c in DROP_HIGH_MISSING if c in df_lc_raw.columns]
df_lc = df_lc_raw.drop(columns=drop_existing)
LOG2.write(f"Sau drop missing>90%: {df_lc.shape}  (loại {len(drop_existing)} cột)")

# ── 2.2.2. Drop leakage / ineffective variables ───────────────────────────────
LOG2.section("2.2.2. Drop leakage / ineffective variables", level=3)
DROP_LEAKAGE = [
    # post-loan / leakage (biết sau khi cho vay)
    "emp_title", "title", "zip_code", "policy_code",
    "out_prncp_inv", "total_pymnt_inv", "recoveries",
    "last_pymnt_d", "last_pymnt_amnt",
    "out_prncp", "total_pymnt", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "collection_recovery_fee",
    "debt_settlement_flag", "hardship_flag", "next_pymnt_d",
    "chargeoff_within_12_mths", "delinq_amnt", "acc_now_delinq",
    "num_tl_120dpd_2m", "num_tl_30dpd",
]
drop_existing2 = [c for c in DROP_LEAKAGE if c in df_lc.columns]
df_lc = df_lc.drop(columns=drop_existing2)
LOG2.write(f"Sau drop leakage: {df_lc.shape}  (loại {len(drop_existing2)} cột)")

# ── 2.2.3. Map loan_status → binary y (1 = default) ──────────────────────────
LOG2.section("2.2.3. Map loan_status → binary y (1 = default)", level=3)
DEFAULT_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}
GOOD_STATUSES    = {"Fully Paid"}
df_lc = df_lc[df_lc["loan_status"].isin(DEFAULT_STATUSES | GOOD_STATUSES)].copy()
df_lc["y"] = df_lc["loan_status"].isin(DEFAULT_STATUSES).astype(int)
df_lc.drop(columns=["loan_status"], inplace=True)
LOG2.write(f"LC sau filter status: {df_lc.shape}, default rate = {df_lc['y'].mean():.4f}")

# ── 2.2.4. Imputation: < 10% missing → median / mode ─────────────────────────
#           10–90% missing  → -99999 (numeric) / "unknown" (categorical)
# (Fit imputation values CHỈ trên tập này — không dùng test set để tính)
LOG2.section("2.2.4. Imputation theo chiến lược R gốc", level=3)
# NOTE: sentinel -99999 cho 10–90% missing là intentional design từ pipeline R gốc.
#       Nó báo hiệu "data không có" một cách explicit cho tree-based models (XGBoost/RF).
#       Với NN (DANN), StandardScaler sẽ map nó thành outlier rất âm — acceptable
#       vì đây là cột missing nhiều, weight của nó trong encoder sẽ nhỏ.
#       Khác với 2.2.8 (NaN do map miss) — ở đó dùng median để không làm lệch phân phối.

missing_pct = df_lc.isnull().mean()

# Cột < 10% missing
cols_low  = missing_pct[(missing_pct > 0) & (missing_pct < 0.10)].index.tolist()
cols_low_num = [c for c in cols_low if df_lc[c].dtype.kind in "fi"]
cols_low_cat = [c for c in cols_low if df_lc[c].dtype.kind not in "fi"]

medians_lc = {c: df_lc[c].median() for c in cols_low_num}
modes_lc   = {c: df_lc[c].mode()[0] for c in cols_low_cat}

for c in cols_low_num:
    df_lc[c] = df_lc[c].fillna(medians_lc[c])
for c in cols_low_cat:
    df_lc[c] = df_lc[c].fillna(modes_lc[c])

# Cột 10–90% missing
cols_high = missing_pct[(missing_pct >= 0.10) & (missing_pct <= 0.90)].index.tolist()
cols_high_num = [c for c in cols_high if df_lc[c].dtype.kind in "fi"]
cols_high_cat = [c for c in cols_high if df_lc[c].dtype.kind not in "fi"]

for c in cols_high_num:
    df_lc[c] = df_lc[c].fillna(-99999)
for c in cols_high_cat:
    df_lc[c] = df_lc[c].fillna("unknown")

# Drop cột còn NaN (missing > 90% nhưng chưa được list ở 2.2.1)
df_lc = df_lc.dropna(axis=1, how="any")
LOG2.write(f"Sau imputation: {df_lc.shape}")

# ── 2.2.5. Binary encoding (5 cột 2 giá trị) ─────────────────────────────────
LOG2.section("2.2.5. Binary encoding", level=3)
if "term" in df_lc.columns:
    df_lc["term"] = (df_lc["term"].astype(str).str.strip() == "60 months").astype(int)
if "pymnt_plan" in df_lc.columns:
    df_lc["pymnt_plan"] = (df_lc["pymnt_plan"].astype(str).str.strip().str.lower() != "n").astype(int)
if "initial_list_status" in df_lc.columns:
    df_lc["initial_list_status"] = (df_lc["initial_list_status"].astype(str).str.strip().str.lower() != "w").astype(int)
if "application_type" in df_lc.columns:
    df_lc["application_type"] = (df_lc["application_type"].astype(str).str.strip().str.lower() != "individual").astype(int)
if "disbursement_method" in df_lc.columns:
    df_lc["disbursement_method"] = (df_lc["disbursement_method"].astype(str).str.strip().str.lower() != "cash").astype(int)
LOG2.write("Binary: term(60m=1), pymnt_plan(!=n=1), initial_list_status(!=w=1), "
           "application_type(!=Individual=1), disbursement_method(!=Cash=1).")

# ── 2.2.6. Label encoding (ordinal: grade, sub_grade, emp_length) ─────────────
LOG2.section("2.2.6. Label encoding ordinal variables", level=3)
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
if "grade" in df_lc.columns:
    df_lc["grade"] = df_lc["grade"].map(GRADE_MAP)

SUBGRADE_LEVELS = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]  # 35 levels
if "sub_grade" in df_lc.columns:
    df_lc["sub_grade"] = df_lc["sub_grade"].map(
        {sg: i+1 for i, sg in enumerate(SUBGRADE_LEVELS)})

EMP_MAP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
           "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
           "8 years": 8, "9 years": 9, "10+ years": 10}
if "emp_length" in df_lc.columns:
    df_lc["emp_length"] = df_lc["emp_length"].map(EMP_MAP)
    df_lc["emp_length"] = df_lc["emp_length"].fillna(-99999)  # "unknown" dạng số

# Xử lý earliest_cr_line → cr_history_yr (số năm lịch sử tín dụng)
if "earliest_cr_line" in df_lc.columns:
    df_lc["earliest_cr_line"] = pd.to_datetime(
        df_lc["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df_lc["cr_history_yr"] = (
        datetime.datetime(2018, 12, 31) - df_lc["earliest_cr_line"]
    ).dt.days / 365.25
    df_lc.drop(columns=["earliest_cr_line"], inplace=True)

# int_rate, revol_util: strip % nếu còn dạng string
for col in ["int_rate", "revol_util"]:
    if col in df_lc.columns and df_lc[col].dtype == object:
        df_lc[col] = df_lc[col].astype(str).str.replace("%", "").astype(float)

LOG2.write("Label encode: grade(1-7), sub_grade(1-35), emp_length(0-10/-99999).")

# ── 2.2.7. One-hot encoding (nominal: addr_state→region, home_ownership, ──────
#                              verification_status, purpose)
LOG2.section("2.2.7. One-hot encoding nominal variables", level=3)
if "addr_state" in df_lc.columns:
    REGION_MAP = {
        **dict.fromkeys(["CT","ME","MA","NH","RI","VT","NJ","NY","PA"], "Northeast"),
        **dict.fromkeys(["IL","IN","MI","OH","WI","IA","KS","MN","MO","NE","ND","SD"], "Midwest"),
        **dict.fromkeys(["DE","FL","GA","MD","NC","SC","VA","WV","DC",
                         "AL","KY","MS","TN","AR","LA","OK","TX"], "South"),
        **dict.fromkeys(["AZ","CO","ID","MT","NV","NM","UT","WY",
                         "AK","CA","HI","OR","WA"], "West"),
    }
    df_lc["region"] = df_lc["addr_state"].map(REGION_MAP).fillna("Unknown")
    df_lc.drop(columns=["addr_state"], inplace=True)

ohe_cols = [c for c in ["home_ownership", "verification_status", "purpose", "region"]
            if c in df_lc.columns]
df_lc = pd.get_dummies(df_lc, columns=ohe_cols, drop_first=True, dummy_na=False)
LOG2.write(f"One-hot: {ohe_cols} (drop_first=True).")
LOG2.write(f"LC sau encode đầy đủ: {df_lc.shape}")

# ── 2.2.8. Finalize: drop object còn sót + xử lý NaN còn lại ─────────────────
# CHIẾN LƯỢC NaN cuối:
#   - Không dùng sentinel -99999 ở bước này vì các cột còn NaN tại đây
#     là do label/grade map không thành công (giá trị ngoài vocab) hoặc
#     là các cột numeric đặc thù chưa được impute ở 2.2.4.
#   - Dùng median impute cho cột numeric (giữ phân phối), mode cho categorical.
#   - Sau đó dropna(axis=0) để loại các hàng còn NaN không thể impute
#     (thường < 0.1% tổng) — an toàn hơn thay bằng -99999 (không lệch phân phối).
LOG2.section("2.2.8. Finalize: drop object còn sót, median-impute NaN, dropna rows", level=3)

# Step 1: drop cột object còn sót sau OHE (không encode được)
obj_cols = [c for c in df_lc.columns if df_lc[c].dtype == object and c != "y"]
if obj_cols:
    LOG2.write(f"  Drop {len(obj_cols)} cột object còn sót: "
               f"{obj_cols[:10]}{'...' if len(obj_cols) > 10 else ''}")
    df_lc.drop(columns=obj_cols, inplace=True)

# Step 2: với các cột numeric còn NaN (ví dụ sub_grade/grade map miss) → median impute
nan_cols = [c for c in df_lc.columns if c != "y" and df_lc[c].isnull().any()]
if nan_cols:
    LOG2.write(f"  Median-impute {len(nan_cols)} cột numeric còn NaN: "
               f"{nan_cols[:10]}{'...' if len(nan_cols) > 10 else ''}")
    for c in nan_cols:
        if df_lc[c].dtype.kind in "fi":
            df_lc[c] = df_lc[c].fillna(df_lc[c].median())

# Step 3: dropna rows — xử lý các hàng vẫn còn NaN sau median (rất hiếm)
n_before = len(df_lc)
df_lc = df_lc.dropna(axis=0).reset_index(drop=True)
n_dropped = n_before - len(df_lc)
LOG2.write(f"  Dropped {n_dropped} rows còn NaN sau impute "
           f"({n_dropped/max(n_before,1)*100:.3f}% tổng).")
LOG2.write(f"LC sau finalize: {df_lc.shape}")

# ── 2.2.9. Stratified subsample 300K ─────────────────────────────────────────
LOG2.section("2.2.9. Stratified subsample 300K", level=3)
y_lc_full = df_lc["y"].values.astype(int)
if len(df_lc) > N_LC_KEEP:
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N_LC_KEEP,
                                 random_state=SEED)
    keep_idx, _ = next(sss.split(np.zeros(len(df_lc)), y_lc_full))
    df_lc = df_lc.iloc[keep_idx].reset_index(drop=True)
LOG2.write(f"LC sau subsample: {df_lc.shape}, default rate = {df_lc['y'].mean():.4f}")

# ── 2.2.10. Tách features / nhãn + StandardScaler ────────────────────────────
y_lc      = df_lc["y"].values.astype(int)
X_lc      = df_lc.drop(columns=["y"]).values.astype(float)
features_lc = [c for c in df_lc.columns if c != "y"]
LOG2.write(f"d_s = {X_lc.shape[1]} (LC), d_t = {X_dc_tr.shape[1]} (DCCC).")

scaler_s    = StandardScaler().fit(X_lc)
X_s_scaled  = scaler_s.transform(X_lc)
scaler_t    = StandardScaler().fit(X_dc_tr)
X_t_tr_scaled = scaler_t.transform(X_dc_tr)
X_t_te_scaled = scaler_t.transform(X_dc_te)

# ──────────────────────────────────────────────────────────────────────────────
# 2.3. HFA core implementation
# ──────────────────────────────────────────────────────────────────────────────
LOG2.section("2.3. HFA core", level=2)

# ── 2.3.1. augment_source / augment_target ───────────────────────────────────
# NOTE: Tổ chức 3 block của vector augment là KEY POINT của HFA. Đảm bảo
#       (i) common subspace có thể so sánh trực tiếp giữa hai miền,
#       (ii) raw features được giữ nguyên (không mất thông tin gốc),
#       (iii) zero-pad đảm bảo block raw chỉ active với miền tương ứng.
def augment_source(X_s, P, d_t):
    """φ_s(x) = [P x ; x ; 0_{d_t}]   (Yang 2020 Eq. 6.8)
    P shape: (d_c, d_s); X_s shape: (n_s, d_s); output: (n_s, d_c+d_s+d_t)."""
    XP   = X_s @ P.T          # (n_s, d_c)
    pad  = np.zeros((X_s.shape[0], d_t))
    return np.hstack([XP, X_s, pad])

def augment_target(X_t, Q, d_s):
    """φ_t(x) = [Q x ; 0_{d_s} ; x]   (Yang 2020 Eq. 6.8)."""
    XQ  = X_t @ Q.T
    pad = np.zeros((X_t.shape[0], d_s))
    return np.hstack([XQ, pad, X_t])

# ── 2.3.2. _init_PQ — Gaussian random projection (JL lemma) ──────────────────
def _init_PQ(d_c, d_s, d_t, seed=SEED):
    """Khởi tạo (P, Q) theo Johnson-Lindenstrauss Gaussian random projection.

    NOTE [Lý thuyết - Achlioptas 2003 Theorem 1.1]:
        Cho R có entry i.i.d. ~ N(0, 1/d_c). Với mọi x:
            Pr[(1-eps)||x||^2 <= ||R x||^2 <= (1+eps)||x||^2]
                >= 1 - 2 exp(-d_c * eps^2 / 8).
        => Random projection bảo toàn norm Euclidean với xác suất cao.

    NOTE [Vì sao d_c=64?]:
        Với eps=0.5, prob bảo toàn ~ 0.73. Với d_c=128 ~ 0.97. Notebook
        chọn 64 cân bằng tốc độ tính và độ tin cậy. r_P=r_Q=10 đảm bảo
        ||P||_F sau init xấp xỉ sqrt(d_c*d_s/d_c) = sqrt(d_s) < 10 thường.

    NOTE [Vì sao không zero-init?]:
        Zero-init làm gradient = 0 ngay bước đầu (Glorot 2010 AISTATS,
        symmetry breaking) => BCD bị stuck.
    """
    rng = np.random.default_rng(seed)
    sigma = 1.0 / np.sqrt(d_c)
    P = rng.normal(0.0, sigma, size=(d_c, d_s))
    Q = rng.normal(0.0, sigma, size=(d_c, d_t))
    return P, Q

# ── 2.3.3. _proj_frobenius_ball — chiếu Euclidean lên cầu Frobenius ─────────
def _proj_frobenius_ball(M, radius):
    """Phép chiếu Euclidean lên cầu Frobenius {M : ||M||_F <= r}.

    NOTE [Lý thuyết - Boyd & Vandenberghe 2004 §8.1.1]:
        Tập C_r = { M : ||M||_F <= r } là lồi đóng. Phép chiếu Euclidean
        có dạng đóng (closed form):
            proj_{C_r}(M) = (r / ||M||_F) * M    nếu ||M||_F > r,
                          = M                     ngược lại.
        Tức là M được rescale nếu vượt biên, không đổi nếu trong biên.

    NOTE [Vì sao Frobenius?]:
        - Phép chiếu closed-form (không cần solver lồi).
        - Tương đương regularizer L2 trên vec(M) - đối ngẫu với Ridge.
        - np.linalg.norm(M) mặc định cho ma trận 2-D = Frobenius norm.
    """
    nrm = np.linalg.norm(M)
    if nrm > radius and nrm > 0:
        return (radius / nrm) * M
    return M

# ── 2.3.4. _grad_PQ — subgradient squared-hinge trên active set ─────────────
def _grad_PQ(P, Q, w, b, X_s, y_s_pm, X_t, y_t_pm, lam_P, lam_Q, C=1.0):
    """Tính subgradient ∂L/∂P, ∂L/∂Q của hàm mục tiêu Eq. 6.8 (Yang 2020).

    NOTE [Lý thuyết — Yang 2020 Ch.6 Eq. 6.8]:
        L(w, b, P, Q) = ½‖w‖² + (λ_P/2)‖P‖_F² + (λ_Q/2)‖Q‖_F²
                       + C · Σ_i ξ_i²,
        với ξ_i = max(0, 1 − y_i z_i) là squared-hinge slack
        và z_i = w·φ(x_i) + b.

    NOTE [Fix D2.1]: thêm hệ số C vào subgradient để khớp với hàm mục tiêu.
        Trước đây coef_s, coef_t = -2·margin·y → thiếu C. Khi C ≠ 1 thì gradient
        w.r.t. (P,Q) sẽ không cùng scale với SVM internal optimization.

    NOTE [Suy diễn]: với squared-hinge ℓ(z) = max(0, 1−y·z)²,
        ∂ℓ/∂z = -2·max(0, 1-y·z)·y trên active set { i : 1-y_i·z_i > 0 }.
        Áp chain rule: z_i = w_c·(P x_s,i) + w_s,raw·x_s,i (block 1+2 của φ_s),
        ⇒ ∂z_i/∂P = y_i · w_c · x_s,i^T (outer product).
        Vậy ∂ℓ_i/∂P = -2·C·(1-y_i z_i)·y_i · w_c · x_s,i^T trên active set.

    NOTE [Active set — Hsieh 2008 ICML]: chỉ tính gradient từ các margin
        violator (i ∈ A), không tính từ điểm well-classified. Tăng tốc và
        không thay đổi optimum.

    Args:
        P, Q       : ma trận chiếu hiện tại (d_c×d_s, d_c×d_t).
        w, b       : tham số SVM trên không gian augment (d_c+d_s+d_t).
        X_s, X_t   : raw features hai miền.
        y_s_pm, y_t_pm : nhãn ở ±1 (đã chuyển từ 0/1 trước khi gọi).
        lam_P, lam_Q   : hệ số regularizer Frobenius (mềm).
        C          : hệ số squared-hinge Eq. 6.8 (mặc định 1.0).

    Returns:
        gP (d_c×d_s), gQ (d_c×d_t).
    """
    d_c, d_s = P.shape
    _,   d_t = Q.shape
    # tách w thành 3 block: w_c (common), w_s (source raw), w_t (target raw)
    w_c = w[:d_c]
    # Augment để tính margin
    Phi_s = augment_source(X_s, P, d_t)
    Phi_t = augment_target(X_t, Q, d_s)
    z_s   = Phi_s @ w + b
    z_t   = Phi_t @ w + b
    margin_s = 1.0 - y_s_pm * z_s
    margin_t = 1.0 - y_t_pm * z_t
    act_s = margin_s > 0
    act_t = margin_t > 0
    # Squared-hinge: ℓ'(z) = -2·max(0, 1−y·z)·y, scale theo C (Eq. 6.8)
    coef_s = -2.0 * C * margin_s * y_s_pm
    coef_t = -2.0 * C * margin_t * y_t_pm
    # ∂L/∂P_ij = Σ_{i ∈ act_s} coef_s_i * w_c[a] * X_s[i,b]
    if act_s.any():
        gP = np.outer(w_c, np.zeros(d_s))
        gP = (coef_s[act_s, None, None]
              * w_c[None, :, None] * X_s[act_s, None, :]).sum(axis=0)
    else:
        gP = np.zeros_like(P)
    if act_t.any():
        gQ = (coef_t[act_t, None, None]
              * w_c[None, :, None] * X_t[act_t, None, :]).sum(axis=0)
    else:
        gQ = np.zeros_like(Q)
    # Cộng đạo hàm regularizer L2 trên P, Q
    gP = gP + lam_P * P
    gQ = gQ + lam_Q * Q
    return gP, gQ

# ── 2.3.5. train_hfa — alternating optimization (BCD) ───────────────────────
def train_hfa(X_s, y_s, X_t, y_t,
              d_c=64, n_outer=5, eta=1e-3,
              C=1.0, lam_P=1e-3, lam_Q=1e-3,
              r_P=10.0, r_Q=10.0,
              verbose=False):
    """Block Coordinate Descent giải Eq. 6.8 (Yang 2020 §6.3.4).

    NOTE [BCD - Tseng 2001 Th. 4.1]:
        Hàm Eq. 6.8 không lồi đồng thời theo (P, Q, w) nhưng lồi theo
        từng khối khi cố định khối khác. BCD luân phiên:

        for k = 1 .. n_outer:
            Bước 1: cố định (P, Q) ⇒ bài toán theo w trở thành SVM tuyến
                    tính trên data augment ⇒ fit LinearSVC ⇒ (w_k, b_k).
            Bước 2: cố định (w_k, b_k) ⇒ projected sub-gradient trên (P, Q).
            Bước 3: chiếu (P, Q) lên cầu Frobenius.

        Tseng 2001 Th. 4.1 đảm bảo mọi điểm tụ là stationary point của
        Eq. 6.8 khi:
            - L lồi theo từng khối,
            - feasible region là tích Cartesian các tập đóng lồi,
            - mỗi bước con không tăng L.

    NOTE [Default & lý do]:
        d_c    = 64       JL eps ~ 0.5 ở prob ~ 0.73.
        n_outer = 5       Đủ converge với dataset trung bình (Duan 2012
                          Table 2 cho thấy 3-10 vòng đều ổn).
        eta    = 1e-3     Đủ nhỏ để projected gradient không overshoot.
        C      = 1.0      Chuẩn LinearSVC.
        lam_P, lam_Q = 1e-3   Frobenius regularizer mềm.
        r_P, r_Q = 10.0   Hard constraint bán kính Frobenius.

    Returns:
        dict {P, Q, w, b, history}.
        history mỗi outer ghi ||P||_F, ||Q||_F, n_active để debug.
    """
    n_s, d_s = X_s.shape
    n_t, d_t = X_t.shape
    P, Q = _init_PQ(d_c, d_s, d_t)
    P = _proj_frobenius_ball(P, r_P)
    Q = _proj_frobenius_ball(Q, r_Q)
    # Convert nhãn 0/1 → ±1 cho hinge
    y_s_pm = np.where(y_s == 1, 1.0, -1.0)
    y_t_pm = np.where(y_t == 1, 1.0, -1.0)
    history = []
    w = b = None
    for outer in range(n_outer):
        # (1) Augment + fit SVM (squared-hinge)
        Phi_s = augment_source(X_s, P, d_t)
        Phi_t = augment_target(X_t, Q, d_s)
        Phi   = np.vstack([Phi_s, Phi_t])
        y_all = np.concatenate([y_s, y_t])
        # NOTE [Fix B2.2]: thêm class_weight='balanced' để khớp với Case 1/2
        # trong ablation (cùng dùng balanced). Trước đây thiếu → Case 3 unfair.
        # Yang 2020 §6.3 Eq. 6.8 không nói rõ class weight, nhưng vì DCCC + LC
        # đều imbalanced (22 % / 14 %) → cần balanced để loss đối xứng giữa
        # class (King & Zeng 2001).
        # NOTE [dual=False]: sklearn docs khuyến nghị dual=False khi
        # n_samples >> n_features (324K >> 187). dual=True yêu cầu giải
        # bài toán dual có n_samples biến → rất chậm. dual=False (primal
        # coordinate descent) phức tạp O(n*d) mỗi iter → nhanh hơn ~10-50x.
        svm = LinearSVC(loss="squared_hinge", C=C, max_iter=2000,
                        dual=False, random_state=SEED,
                        class_weight="balanced")
        svm.fit(Phi, y_all)
        w = svm.coef_.ravel().astype(float)
        b = float(svm.intercept_[0])
        # (2) Subgradient step trên (P, Q)
        # NOTE: truyền C để gradient w.r.t. (P,Q) đồng nhất với SVM internal.
        gP, gQ = _grad_PQ(P, Q, w, b, X_s, y_s_pm, X_t, y_t_pm,
                          lam_P, lam_Q, C=C)
        P = P - eta * gP
        Q = Q - eta * gQ
        # (3) Frobenius projection
        P = _proj_frobenius_ball(P, r_P)
        Q = _proj_frobenius_ball(Q, r_Q)
        # NOTE [Fix B2.1]: tính n_active = số sample violating margin
        # (margin_i = 1 - y_i*z_i > 0). Trước đây code dùng
        # int(boolarray).sum() → TypeError. Đã sửa thành
        # int((boolarray).sum()) — đúng numpy semantics.
        y_all_pm = np.concatenate([y_s_pm, y_t_pm])
        z_all    = Phi @ w + b
        n_active = int(((1.0 - y_all_pm * z_all) > 0).sum())
        history.append({"outer"   : outer + 1,
                        "||P||_F" : float(np.linalg.norm(P)),
                        "||Q||_F" : float(np.linalg.norm(Q)),
                        "n_active": n_active})
        if verbose:
            print(f"   [HFA outer {outer+1}/{n_outer}] "
                  f"||P||_F={history[-1]['||P||_F']:.3f}  "
                  f"||Q||_F={history[-1]['||Q||_F']:.3f}")
    return {"P": P, "Q": Q, "w": w, "b": b, "history": history}

# ── 2.3.6. predict_hfa + Platt calibration ──────────────────────────────────
def predict_hfa_score(model, X_t):
    """Score = w·φ_t(X_t) + b (tuyến tính trong augmented space)."""
    d_s = model["P"].shape[1]
    d_t = model["Q"].shape[1]
    Phi_t = augment_target(X_t, model["Q"], d_s)
    return Phi_t @ model["w"] + model["b"]

def _platt_calibrate(scores_train, y_train):
    """Hiệu chuẩn xác suất theo Platt 1999 §5.

    NOTE [Lý thuyết - Platt 1999]:
        SVM cho decision score f(x) = w . phi(x) + b ∈ R - không phải
        xác suất. Platt scaling fit:
            P_hat(y=1|x) = sigma(A * f(x) + B) = 1/(1 + exp(-A f - B))
        bằng MLE trên (f(x_i), y_i). Sklearn LogisticRegression 1-feature
        input ≡ MLE Platt.

    NOTE [C=1e6]: regularization rất nhẹ ~ MLE thuần.
    NOTE [class_weight='balanced']: bù imbalanced (King & Zeng 2001).

    NOTE [Cải tiến tiềm năng]:
        Platt khuyến nghị fit calibration trên fold KHÁC fold fit SVM
        để tránh over-confident calibration. Có thể thay bằng
        CalibratedClassifierCV(method='sigmoid', cv=3) cho chuẩn hơn.
    """
    lr = LogisticRegression(C=1e6, solver="lbfgs",
                             class_weight="balanced", max_iter=200)
    lr.fit(scores_train.reshape(-1, 1), y_train)
    A = float(lr.coef_[0, 0])
    B = float(lr.intercept_[0])
    return A, B

def predict_hfa_proba(model, X_t, A, B):
    s = predict_hfa_score(model, X_t)
    return 1.0 / (1.0 + np.exp(-(A * s + B)))


# ──────────────────────────────────────────────────────────────────────────────
# 2.4. 5-fold CV trên DCCC (LC làm source cố định)
# ──────────────────────────────────────────────────────────────────────────────
LOG2.section("2.4. HFA — 5-fold Cross-Validation", level=2)
LOG_CV2 = TeeLogger(OUTPUT_DIR / "02a_hfa_cv.txt")
LOG_CV2.section("PHẦN 2 — HFA — CHI TIẾT 5-FOLD CV", level=1)

# Dùng toàn bộ 300K source đã subsample ở bước 2.2.6 (thống nhất với hold-out)
N_LC_CV = min(300_000, X_s_scaled.shape[0])
rng_cv = np.random.default_rng(SEED)
idx_lc_cv = rng_cv.choice(X_s_scaled.shape[0], size=N_LC_CV, replace=False)
X_s_cv = X_s_scaled[idx_lc_cv]
y_s_cv = y_lc[idx_lc_cv]
LOG_CV2.write(f"Source CV size: {X_s_cv.shape}")
LOG_CV2.write(f"Target CV pool: {X_t_tr_scaled.shape}")

cv_iter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
hfa_cv_scores = []
for fold, (tr_idx, va_idx) in enumerate(cv_iter.split(X_t_tr_scaled, y_dc_tr), 1):
    X_tr = X_t_tr_scaled[tr_idx]; y_tr = y_dc_tr[tr_idx]
    X_va = X_t_tr_scaled[va_idx]; y_va = y_dc_tr[va_idx]
    LOG_CV2.section(f"Fold {fold}/{N_FOLDS}", level=3)
    model = train_hfa(X_s_cv, y_s_cv, X_tr, y_tr,
                       d_c=64, n_outer=5, eta=1e-3,
                       C=1.0, r_P=10.0, r_Q=10.0,
                       verbose=True)  # hiện tiến độ từng outer iter
    s_va = predict_hfa_score(model, X_va)
    s_tr = predict_hfa_score(model, X_tr)
    A, B = _platt_calibrate(s_tr, y_tr)
    p_va = 1.0 / (1.0 + np.exp(-(A * s_va + B)))
    p_tr = 1.0 / (1.0 + np.exp(-(A * s_tr + B)))
    thr_tr, _ = find_optimal_threshold_ks(y_tr, p_tr)
    metrics = evaluate(f"HFA (fold {fold})", y_va, p_va, threshold=thr_tr)
    hfa_cv_scores.append(metrics)
    LOG_CV2.write(fmt_metrics_row(metrics))
    LOG2.write(f"  Fold {fold}: {fmt_metrics_row(metrics)}")

# Aggregate
auc_mean = np.mean([m["AUC"] for m in hfa_cv_scores])
ks_mean  = np.mean([m["KS"]  for m in hfa_cv_scores])
auc_std  = np.std([m["AUC"]  for m in hfa_cv_scores])
ks_std   = np.std([m["KS"]   for m in hfa_cv_scores])
LOG_CV2.section("Tổng hợp 5-fold", level=3)
LOG_CV2.write(f"AUC = {auc_mean:.4f} ± {auc_std:.4f}")
LOG_CV2.write(f"KS  = {ks_mean:.4f} ± {ks_std:.4f}")
LOG_CV2.close()
LOG2.write(f"Trung bình 5-fold: AUC={auc_mean:.4f}±{auc_std:.4f}, "
           f"KS={ks_mean:.4f}±{ks_std:.4f}")
LOG2.write(f"Chi tiết → {OUTPUT_DIR}/02a_hfa_cv.txt")

# ──────────────────────────────────────────────────────────────────────────────
# 2.5. Hold-out evaluation + 4-case ablation
# ──────────────────────────────────────────────────────────────────────────────
LOG2.section("2.5. HFA — Hold-out + 3-case Ablation", level=2)
LOG_HO2 = TeeLogger(OUTPUT_DIR / "02b_hfa_holdout.txt")
LOG_HO2.section("PHẦN 2 — HFA — HOLD-OUT (full-train)", level=1)
LOG_AB2 = TeeLogger(OUTPUT_DIR / "02c_hfa_ablation.txt")
LOG_AB2.section("PHẦN 2 — HFA — 3-CASE ABLATION", level=1)

# Helper: chạy SVM thuần trên target (case 1) hoặc trên (concat naive — case 2)
#         hoặc trên HFA augment (case 3).
def _svm_fit_predict(X_tr, y_tr, X_te):
    # dual=False: n_samples (300K+) >> n_features → primal nhanh hơn
    svm = LinearSVC(loss="squared_hinge", C=1.0, max_iter=2000, dual=False,
                    random_state=SEED, class_weight="balanced")
    svm.fit(X_tr, y_tr)
    s_tr = X_tr @ svm.coef_.ravel() + svm.intercept_[0]
    s_te = X_te @ svm.coef_.ravel() + svm.intercept_[0]
    A, B = _platt_calibrate(s_tr, y_tr)
    p_tr = 1.0 / (1.0 + np.exp(-(A * s_tr + B)))
    p_te = 1.0 / (1.0 + np.exp(-(A * s_te + B)))
    thr_tr, _ = find_optimal_threshold_ks(y_tr, p_tr)
    return p_te, thr_tr

ablation_results = []

# --- Case 1: DCCC alone (no transfer)
LOG_AB2.section("Case 1: DCCC alone (LinearSVC trên target only)", level=2)
p_te, thr = _svm_fit_predict(X_t_tr_scaled, y_dc_tr, X_t_te_scaled)
m = evaluate("HFA Case 1 (DCCC alone)", y_dc_te, p_te, threshold=thr)
ablation_results.append(m); LOG_AB2.write(fmt_metrics_row(m))

# --- Case 2: Concat naive — chỉ khả thi nếu cùng dim. Vì heterogeneous,
#     ta xếp 2 tập thành block-diagonal augment đơn giản: [LC ; 0] vs [0 ; DCCC]
LOG_AB2.section("Case 2: Concat naive (block-diag augment, không HFA)", level=2)
d_s = X_s_cv.shape[1]; d_t = X_t_tr_scaled.shape[1]
Phi_s_naive = np.hstack([X_s_cv,    np.zeros((X_s_cv.shape[0],  d_t))])
Phi_t_naive = np.hstack([np.zeros((X_t_tr_scaled.shape[0], d_s)), X_t_tr_scaled])
Phi_te_naive= np.hstack([np.zeros((X_t_te_scaled.shape[0], d_s)), X_t_te_scaled])
p_te, thr = _svm_fit_predict(
    np.vstack([Phi_s_naive, Phi_t_naive]),
    np.concatenate([y_s_cv, y_dc_tr]),
    Phi_te_naive)
m = evaluate("HFA Case 2 (Concat naive)", y_dc_te, p_te, threshold=thr)
ablation_results.append(m); LOG_AB2.write(fmt_metrics_row(m))

# --- Case 3: HFA full method
LOG_AB2.section("Case 3: HFA full method (LC source → DCCC target)", level=2)
hfa_model = train_hfa(X_s_cv, y_s_cv, X_t_tr_scaled, y_dc_tr,
                      d_c=64, n_outer=5, eta=1e-3,
                      C=1.0, r_P=10.0, r_Q=10.0)
s_tr = predict_hfa_score(hfa_model, X_t_tr_scaled)
s_te = predict_hfa_score(hfa_model, X_t_te_scaled)
A, B = _platt_calibrate(s_tr, y_dc_tr)
p_tr = 1.0 / (1.0 + np.exp(-(A * s_tr + B)))
p_te = 1.0 / (1.0 + np.exp(-(A * s_te + B)))
thr_tr, _ = find_optimal_threshold_ks(y_dc_tr, p_tr)
m = evaluate("HFA Case 3 (Full HFA)", y_dc_te, p_te, threshold=thr_tr)
ablation_results.append(m); LOG_AB2.write(fmt_metrics_row(m))
LOG_HO2.write(fmt_metrics_row(m))
LOG_HO2.write(f"||P||_F = {np.linalg.norm(hfa_model['P']):.4f}")
LOG_HO2.write(f"||Q||_F = {np.linalg.norm(hfa_model['Q']):.4f}")
LOG_HO2.write(f"History (n_outer iterations):")
for h in hfa_model["history"]:
    LOG_HO2.write(f"  outer {h['outer']}: |P|={h['||P||_F']:.3f} |Q|={h['||Q||_F']:.3f}")


LOG_AB2.section("Bảng tổng hợp 3-case ablation", level=2)
hdr = f"  {'Case':<25s} {'AUC':>8s} {'KS':>8s} {'Brier':>8s} {'Sens':>8s} {'Spec':>8s}"
LOG_AB2.write(hdr); LOG_AB2.write("  " + "-" * 70)
for r in ablation_results:
    LOG_AB2.write(f"  {r['Model']:<25s} {r['AUC']:>8.4f} {r['KS']:>8.4f} "
                  f"{r['Brier']:>8.4f} {r['Sens']:>8.4f} {r['Spec']:>8.4f}")
LOG_AB2.close(); LOG_HO2.close()
LOG2.write(f"Chi tiết hold-out → {OUTPUT_DIR}/02b_hfa_holdout.txt")
LOG2.write(f"Chi tiết ablation → {OUTPUT_DIR}/02c_hfa_ablation.txt")

# ──────────────────────────────────────────────────────────────────────────────
# 2.6. Tóm tắt phần HFA
# ──────────────────────────────────────────────────────────────────────────────
LOG2.section("2.6. Tóm tắt phần HFA", level=2)
LOG2.write(f"5-fold CV: AUC={auc_mean:.4f}±{auc_std:.4f}, "
           f"KS={ks_mean:.4f}±{ks_std:.4f}")
LOG2.write("Hold-out 3-case:")
for r in ablation_results:
    LOG2.write(f"  {r['Model']:<25s} | {fmt_metrics_row(r)}")

hfa_holdout_main = ablation_results[2]   # Case 3 = HFA full
LOG2.close()
RUN_LOG.write(f"[Done] PHẦN 2 — HFA → {OUTPUT_DIR}/02_hfa.txt")


# ════════════════════════════════════════════════════════════════════════
# PHẦN 3 — DANN (Domain-Adversarial Neural Network)
# ════════════════════════════════════════════════════════════════════════
#
# NGUỒN LÝ THUYẾT BÁM SÁT:
#   - Ganin Y., Lempitsky V. (2015) "Unsupervised Domain Adaptation by
#     Backpropagation", ICML — bài báo gốc DANN.
#   - Ganin Y. et al. (2016) "Domain-Adversarial Training of Neural
#     Networks", JMLR 17:1-35 — phiên bản mở rộng đầy đủ.
#   - Ben-David S. et al. (2010) "A theory of learning from different
#     domains", Mach. Learn. 79:151-175 — H-divergence bound.
#   - Goodfellow I. et al. (2014) NeurIPS — Generative Adversarial Networks
#     (nguyên lý min-max gốc).
#   - He K. et al. (2015) ICCV — Kaiming initialization cho ReLU.
#   - Ba J. et al. (2016) arXiv:1607.06450 — Layer Normalization.
#   - Kingma D., Ba J. (2015) ICLR — Adam optimizer.
#   - Pascanu R. et al. (2013) ICML — Gradient clipping.
#
# LÝ THUYẾT (Ben-David 2010 Th. 2):
#   epsilon_T(h) <= epsilon_S(h) + (1/2) d_{H△H}(D_S, D_T) + lambda*
#
#   ⇒ Để giảm rủi ro target, ta giảm H-divergence d_{H△H} giữa source
#   và target trong không gian biểu diễn (representation space).
#
# IDEA DANN (Ganin 2016):
#   Học một encoder g sao cho domain classifier d KHÔNG phân biệt được
#   source vs target trong g-space, thông qua MIN-MAX:
#       min_{g, c} max_{d}  L_label(c(g(x_s)), y_s)
#                          - lambda * L_domain(d(g(x)), domain_label)
#   Cài đặt qua Gradient Reversal Layer (GRL):
#       forward: identity
#       backward: -lambda * grad
#   ⇒ Trong một backward pass duy nhất, encoder MAX domain loss (lừa d)
#   trong khi d MIN domain loss (đối nghịch nhau).
#
# KIẾN TRÚC HeteroDANN (cho d_s != d_t):
#   - DomainEncoder_S: R^{d_s} -> R^{hidden}
#   - DomainEncoder_T: R^{d_t} -> R^{hidden}
#     (hai encoder không chia sẻ trọng số vì input shape khác)
#   - LabelPredictor:  R^{hidden} -> R (logit binary)
#   - DomainClassifier (qua GRL): R^{hidden} -> R (logit domain)
#
# NOTE QUAN TRỌNG [Tuân thủ Ganin 2016 gốc]:
#   Ganin 2016 là UNSUPERVISED DA - KHÔNG dùng nhãn target cho training.
#   Chỉ dùng source labels cho classification loss và domain labels
#   (source=0, target=1) cho domain adversarial loss.
#   Nhãn target chỉ dùng ĐỂ ĐÁNH GIÁ sau huấn luyện.
#
# ════════════════════════════════════════════════════════════════════════
LOG3 = TeeLogger(OUTPUT_DIR / "03_dann.txt")
LOG3.section("PHẦN 3 — DANN: Domain-Adversarial Neural Network", level=1)
LOG3.write("Tham chiếu: Ganin Y. & Lempitsky V. (2015) ICML; Ganin Y. et al. "
           "(2016) JMLR 17:1–35; Ben-David S. et al. (2010) Mach. Learn. "
           "79:151–175 (H-divergence bound).")

# ──────────────────────────────────────────────────────────────────────────────
# 3.1. Kiến trúc HeteroDANN
# ──────────────────────────────────────────────────────────────────────────────
LOG3.section("3.1. Kiến trúc HeteroDANN", level=2)

# ── 3.1.1. GradientReversalFn ────────────────────────────────────────────────
class GradientReversalFn(torch.autograd.Function):
    """Gradient Reversal Layer - core trick của DANN.

    NOTE [Ganin & Lempitsky 2015 ICML §3.2]:
        forward(x)  = x                  (identity)
        backward(g) = -lambda * g        (đảo dấu, scale theo lambda)

    NOTE [Vì sao cần GRL]:
        Min-max của adversarial training thường cần two-step optimization
        (alternate update generator vs discriminator). GRL cho phép
        một-pass: gradient ngược ra encoder bị đảo dấu => encoder MAX
        domain loss (= lừa domain classifier), trong khi domain
        classifier vẫn MIN domain loss (giải thẳng).

    NOTE [Lý do x.view_as(x)]:
        PyTorch yêu cầu output của Function phải là tensor mới (không
        phải alias) để autograd track đúng (xem PyTorch issue #6020).
        view_as(x) tạo view mới cùng dữ liệu nhưng tensor object mới.

    NOTE [None thứ hai trong return của backward]:
        lambda_ là hyperparameter, không cần gradient => trả None.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    """Wrapping module để dễ tích hợp nn.Sequential và device placement.

    NOTE: lambda_ được cập nhật mỗi step (không phải hyperparameter cố
    định) - thay đổi qua set_lambda() từ training loop theo lịch
    Ganin 2016: lambda(p) = 2/(1+exp(-10p)) - 1.
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def set_lambda(self, lambda_):
        self.lambda_ = float(lambda_)
    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambda_)

# ── 3.1.2. DomainEncoder ─────────────────────────────────────────────────────
class DomainEncoder(nn.Module):
    """MLP encoder cho mỗi domain.

    NOTE [Kiến trúc]: Linear -> LayerNorm -> ReLU -> Dropout
                   -> Linear -> LayerNorm -> ReLU
        (2 layers, hidden=128, dropout=0.2 mặc định).

    NOTE [Vì sao LayerNorm thay BatchNorm — Ba 2016]:
        Trong DANN, mỗi mini-batch chứa cả source và target sample.
        Nếu BN, statistics bị tính trên hỗn hợp domain ⇒ smear
        distribution boundary ⇒ encoder khó học domain-invariant.
        LN tính theo feature dim của TỪNG sample ⇒ batch-independent.

    NOTE [Kaiming init - He 2015 ICCV §2.2]:
        ReLU triệt nửa input ⇒ Glorot init bị under-scaled cho ReLU.
        Kaiming dùng sigma^2 = 2/fan_in (thay 1/fan_in của Xavier)
        để bù cho ReLU. Bias init = 0 chuẩn.

    NOTE [Hai encoder riêng]: do d_s != d_t (heterogeneous setting),
        không thể chia sẻ trọng số layer đầu. Hai encoder hoàn toàn
        độc lập, chỉ chia sẻ output dim = hidden để cùng vào
        LabelPredictor và DomainClassifier.
    """
    def __init__(self, in_dim, hidden=128, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming init cho ReLU (He et al. 2015 ICCV)
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

# ── 3.1.3. LabelPredictor ────────────────────────────────────────────────────
class LabelPredictor(nn.Module):
    """h → logit nhị phân (binary default Y/N)."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, h):
        return self.net(h).squeeze(-1)

# ── 3.1.4. DomainClassifier ──────────────────────────────────────────────────
class DomainClassifier(nn.Module):
    """h → logit domain (0 = source, 1 = target). Đặt SAU GRL."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, h):
        return self.net(h).squeeze(-1)

# ── 3.1.5. HeteroDANN — assembled model ──────────────────────────────────────
class HeteroDANN(nn.Module):
    """Hai encoder riêng cho source/target (heterogeneous), shared label
    predictor và domain classifier (qua GRL)."""
    def __init__(self, d_s, d_t, hidden=128, p_drop=0.2):
        super().__init__()
        self.encoder_s = DomainEncoder(d_s, hidden, p_drop)
        self.encoder_t = DomainEncoder(d_t, hidden, p_drop)
        self.label     = LabelPredictor(hidden)
        self.grl       = GradientReversalLayer(lambda_=0.0)
        self.domain    = DomainClassifier(hidden)
    def forward_source(self, x_s):
        h_s = self.encoder_s(x_s)
        return h_s, self.label(h_s), self.domain(self.grl(h_s))
    def forward_target(self, x_t):
        h_t = self.encoder_t(x_t)
        return h_t, self.label(h_t), self.domain(self.grl(h_t))

LOG3.write("3.1.1. GradientReversalFn — forward identity, backward −λ·∇.")
LOG3.write("3.1.2. DomainEncoder — MLP riêng cho mỗi domain (do d_s≠d_t).")
LOG3.write("3.1.3. LabelPredictor — head dự đoán default.")
LOG3.write("3.1.4. DomainClassifier — head phân biệt source/target qua GRL.")
LOG3.write("3.1.5. HeteroDANN — gói chung hai encoder + 2 head.")

# ──────────────────────────────────────────────────────────────────────────────
# 3.2. Lịch λ và lịch LR
# ──────────────────────────────────────────────────────────────────────────────
LOG3.section("3.2. Lịch trình", level=2)

# ── 3.2.1. compute_lambda(p) ─────────────────────────────────────────────────
def compute_lambda(step, total_steps, gamma=10.0):
    """λ(p) = 2/(1+exp(−γp)) − 1. Ganin 2016 §4.2."""
    p = step / max(1, total_steps)
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)

# ── 3.2.2. compute_lr(p) ─────────────────────────────────────────────────────
def compute_lr(initial_lr, step, total_steps, alpha=10.0, beta=0.75):
    """η(p) = η_0 / (1 + α p)^β. Ganin 2016 §4.2."""
    p = step / max(1, total_steps)
    return float(initial_lr / (1.0 + alpha * p) ** beta)

LOG3.write("3.2.1. compute_lambda(p) = 2/(1+e^{-10p}) - 1   (warm-up adversarial)")
LOG3.write("3.2.2. compute_lr(p)     = 0.01/(1+10p)^0.75   (polynomial decay)")
LOG3.write("Tham số: total_steps=4000, batch_size=64, hidden=128, initial_lr=0.01")

# ──────────────────────────────────────────────────────────────────────────────
# 3.3. train_dann — vòng lặp huấn luyện
# ──────────────────────────────────────────────────────────────────────────────
LOG3.section("3.3. train_dann()", level=2)

def _to_tensor(X, dtype=torch.float32):
    return torch.tensor(np.asarray(X), dtype=dtype, device=DEVICE)

def train_dann(model, X_s, y_s, X_t, y_t=None,
               total_steps=4000, batch_size=64, initial_lr=0.01,
               max_grad_norm=1.0, verbose=False):
    """Step-based training cho HeteroDANN (Ganin 2016 §4.2).

    NOTE [QUAN TRỌNG — Unsupervised DA theo Ganin 2016]:
        Ganin 2016 là UNSUPERVISED Domain Adaptation: chỉ dùng nhãn
        SOURCE cho classification loss. Target KHÔNG có nhãn trong
        bài toán gốc. Nhãn target (nếu có) KHÔNG được dùng trong
        training loop — chỉ dùng để đánh giá SAU huấn luyện.

    NOTE [Loss tổng — Ganin 2016 Eq. 1-4]:
        L = BCE_s(pos_weight=w_s)       ← CHỈ source classification
          + BCE_d                        ← domain loss (source + target)
        (GRL nằm GIỮA encoder và domain classifier nên gradient
         của BCE_d lan xuống encoder bị đảo dấu và scale theo lambda)

    NOTE [Class weighting]:
        - pos_weight_S = (#neg_S) / (#pos_S) ≈ 6.14 (LC 14% default).
        ⇒ Bù imbalanced cho source classification loss.

    NOTE [Lịch lambda - Ganin 2016 §4.2]:
        lambda(p) = 2/(1+exp(-10p)) - 1, p = step/total_steps.
        - lambda(0)   = 0       (warm-up: chưa adversarial)
        - lambda(0.5) ≈ 0.987   (lên nhanh)
        - lambda(1)   ≈ 1       (saturate)
        Warm-up tránh disrupt encoder trước khi có representation tốt.

    NOTE [Lịch LR - Ganin 2016 §4.2]:
        eta(p) = eta_0 / (1 + 10p)^0.75. Polynomial decay - chậm hơn
        exp decay nhưng nhanh hơn 1/sqrt(t).

    NOTE [Adam - Kingma & Ba 2015]:
        beta_1=0.9, beta_2=0.999, eps=1e-8 (mặc định). Adam ổn định
        cho adversarial training (gradient có thể đột ngột do GRL).

    NOTE [Gradient clipping - Pascanu 2013 ICML]:
        max_norm = 1.0 (nhỏ - khuyến nghị Goodfellow 2014 cho GAN).
        GRL có thể đảo một large gradient ⇒ clip giới hạn rủi ro.
    """
    model.to(DEVICE)
    n_s = X_s.shape[0]; n_t = X_t.shape[0]
    pos_w_s = torch.tensor(
        [(y_s == 0).sum() / max((y_s == 1).sum(), 1)],
        dtype=torch.float32, device=DEVICE)

    bce_s = nn.BCEWithLogitsLoss(pos_weight=pos_w_s)
    bce_d = nn.BCEWithLogitsLoss()

    opt = optim.Adam(model.parameters(), lr=initial_lr,
                     betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(SEED)
    history = []

    Xs_t = _to_tensor(X_s); ys_t = _to_tensor(y_s)
    Xt_t = _to_tensor(X_t)

    model.train()
    for step in range(1, total_steps + 1):
        # (1) Cập nhật λ và LR
        lam = compute_lambda(step, total_steps)
        lr  = compute_lr(initial_lr, step, total_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        model.grl.set_lambda(lam)

        # (2) Sample batch source và batch target
        idx_s = rng.integers(0, n_s, size=batch_size)
        idx_t = rng.integers(0, n_t, size=batch_size)
        xb_s = Xs_t[idx_s]; yb_s = ys_t[idx_s]
        xb_t = Xt_t[idx_t]

        # (3) Forward source và target
        _, logit_s, dom_s = model.forward_source(xb_s)
        _, _,       dom_t = model.forward_target(xb_t)

        # (4) Loss — Ganin 2016: CHỈ source label loss + domain loss
        # Target label KHÔNG được dùng (unsupervised DA)
        L_lab_s = bce_s(logit_s, yb_s)
        dom_lbl = torch.cat([torch.zeros_like(dom_s),
                              torch.ones_like(dom_t)])
        L_dom   = bce_d(torch.cat([dom_s, dom_t]), dom_lbl)
        L = L_lab_s + L_dom

        # (5) Backward + clip + step
        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        if verbose and step % 200 == 0:
            history.append({
                "step": step,
                "lambda": round(lam, 4), "lr": round(lr, 5),
                "L_lab_s": float(L_lab_s.item()),
                "L_dom":   float(L_dom.item()),
            })
    return history

# ──────────────────────────────────────────────────────────────────────────────
# 3.4. predict_dann + evaluate
# ──────────────────────────────────────────────────────────────────────────────
def predict_dann(model, X_t, batch=512):
    """Dùng encoder_t + label predictor (không qua GRL/domain)."""
    model.eval()
    Xt_t = _to_tensor(X_t)
    out = []
    with torch.no_grad():
        for i in range(0, Xt_t.shape[0], batch):
            h = model.encoder_t(Xt_t[i:i+batch])
            logit = model.label(h)
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)

# ──────────────────────────────────────────────────────────────────────────────
# 3.5. 5-fold CV trên DCCC
# ──────────────────────────────────────────────────────────────────────────────
LOG3.section("3.5. DANN — 5-fold Cross-Validation", level=2)
LOG_CV3 = TeeLogger(OUTPUT_DIR / "03a_dann_cv.txt")
LOG_CV3.section("PHẦN 3 — DANN — CHI TIẾT 5-FOLD CV", level=1)

dann_cv_scores = []
cv_iter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_idx, va_idx) in enumerate(cv_iter.split(X_t_tr_scaled, y_dc_tr), 1):
    LOG_CV3.section(f"Fold {fold}/{N_FOLDS}", level=3)
    X_tr = X_t_tr_scaled[tr_idx]; y_tr = y_dc_tr[tr_idx]
    X_va = X_t_tr_scaled[va_idx]; y_va = y_dc_tr[va_idx]
    model = HeteroDANN(d_s=X_s_cv.shape[1], d_t=X_tr.shape[1],
                        hidden=128, p_drop=0.2)
    train_dann(model, X_s_cv, y_s_cv, X_tr,
               total_steps=4000, batch_size=64, initial_lr=0.01)
    p_va = predict_dann(model, X_va)
    p_tr = predict_dann(model, X_tr)
    thr_tr, _ = find_optimal_threshold_ks(y_tr, p_tr)
    metrics = evaluate(f"DANN (fold {fold})", y_va, p_va, threshold=thr_tr)
    dann_cv_scores.append(metrics)
    LOG_CV3.write(fmt_metrics_row(metrics))
    LOG3.write(f"  Fold {fold}: {fmt_metrics_row(metrics)}")

dann_auc_mean = np.mean([m["AUC"] for m in dann_cv_scores])
dann_auc_std  = np.std([m["AUC"]  for m in dann_cv_scores])
dann_ks_mean  = np.mean([m["KS"]  for m in dann_cv_scores])
dann_ks_std   = np.std([m["KS"]   for m in dann_cv_scores])
LOG_CV3.section("Tổng hợp 5-fold", level=3)
LOG_CV3.write(f"AUC = {dann_auc_mean:.4f} ± {dann_auc_std:.4f}")
LOG_CV3.write(f"KS  = {dann_ks_mean:.4f} ± {dann_ks_std:.4f}")
LOG_CV3.close()
LOG3.write(f"Trung bình 5-fold: AUC={dann_auc_mean:.4f}±{dann_auc_std:.4f}, "
           f"KS={dann_ks_mean:.4f}±{dann_ks_std:.4f}")
LOG3.write(f"Chi tiết → {OUTPUT_DIR}/03a_dann_cv.txt")

# ──────────────────────────────────────────────────────────────────────────────
# 3.6. Hold-out + 4-case Ablation
# ──────────────────────────────────────────────────────────────────────────────
LOG3.section("3.6. DANN — Hold-out + 3-case Ablation", level=2)
LOG_HO3 = TeeLogger(OUTPUT_DIR / "03b_dann_holdout.txt")
LOG_HO3.section("PHẦN 3 — DANN — HOLD-OUT (full-train)", level=1)
LOG_AB3 = TeeLogger(OUTPUT_DIR / "03c_dann_ablation.txt")
LOG_AB3.section("PHẦN 3 — DANN — 3-CASE ABLATION", level=1)

dann_ablation = []

# --- Case 1: DCCC alone (chỉ encoder_t + label, không adversarial)
LOG_AB3.section("Case 1: DCCC alone (chỉ encoder_t, λ=0)", level=2)
class _NoAdvDANN(HeteroDANN):
    """λ luôn = 0 → không adversarial signal."""
    pass

model = _NoAdvDANN(d_s=X_s_cv.shape[1], d_t=X_t_tr_scaled.shape[1],
                   hidden=128, p_drop=0.2)
# Train chỉ trên target (bỏ qua source pass bằng cách fake không cập nhật từ source)
# Cách thuần: train trên bce_t với pos_weight, không có domain loss
def _train_target_only(model, X_t, y_t, total_steps=4000, batch_size=64, lr=0.01):
    model.to(DEVICE).train()
    Xt_t = _to_tensor(X_t); yt_t = _to_tensor(y_t)
    pos_w = torch.tensor(
        [(y_t == 0).sum() / max((y_t == 1).sum(), 1)],
        dtype=torch.float32, device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)
    for step in range(1, total_steps+1):
        for g in opt.param_groups:
            g["lr"] = compute_lr(lr, step, total_steps)
        idx = rng.integers(0, X_t.shape[0], size=batch_size)
        xb = Xt_t[idx]; yb = yt_t[idx]
        h = model.encoder_t(xb)
        logit = model.label(h)
        L = bce(logit, yb)
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

_train_target_only(model, X_t_tr_scaled, y_dc_tr, total_steps=4000, batch_size=64)
p_te = predict_dann(model, X_t_te_scaled)
p_tr = predict_dann(model, X_t_tr_scaled)
thr_tr, _ = find_optimal_threshold_ks(y_dc_tr, p_tr)
m = evaluate("DANN Case 1 (DCCC alone)", y_dc_te, p_te, threshold=thr_tr)
dann_ablation.append(m); LOG_AB3.write(fmt_metrics_row(m))

# --- Case 2: Concat naive — không thể vì d_s ≠ d_t. Thay bằng "union-feature":
# pad nhãn vào source padded zeros lên d_t và ngược lại — tương đương HFA case 2
# nhưng cho NN. Đơn giản: chia sẻ một encoder duy nhất sau khi pad source về d_t.
LOG_AB3.section("Case 2: Concat naive (pad source về chiều target)", level=2)
if X_s_cv.shape[1] >= X_t_tr_scaled.shape[1]:
    X_s_pad = X_s_cv[:, :X_t_tr_scaled.shape[1]]   # cắt source
else:
    pad_w = X_t_tr_scaled.shape[1] - X_s_cv.shape[1]
    X_s_pad = np.hstack([X_s_cv, np.zeros((X_s_cv.shape[0], pad_w))])
X_concat = np.vstack([X_s_pad, X_t_tr_scaled])
y_concat = np.concatenate([y_s_cv, y_dc_tr])
model_naive = HeteroDANN(d_s=X_t_tr_scaled.shape[1], d_t=X_t_tr_scaled.shape[1],
                          hidden=128, p_drop=0.2)
_train_target_only(model_naive, X_concat, y_concat, total_steps=4000, batch_size=64)
p_te = predict_dann(model_naive, X_t_te_scaled)
p_tr = predict_dann(model_naive, X_t_tr_scaled)
thr_tr, _ = find_optimal_threshold_ks(y_dc_tr, p_tr)
m = evaluate("DANN Case 2 (Concat naive)", y_dc_te, p_te, threshold=thr_tr)
dann_ablation.append(m); LOG_AB3.write(fmt_metrics_row(m))

# --- Case 3: DANN full method (HeteroDANN với GRL, λ schedule)
LOG_AB3.section("Case 3: DANN full method", level=2)
model_full = HeteroDANN(d_s=X_s_cv.shape[1], d_t=X_t_tr_scaled.shape[1],
                         hidden=128, p_drop=0.2)
hist = train_dann(model_full, X_s_cv, y_s_cv, X_t_tr_scaled,
                  total_steps=4000, batch_size=64, initial_lr=0.01,
                  verbose=True)
p_te = predict_dann(model_full, X_t_te_scaled)
p_tr = predict_dann(model_full, X_t_tr_scaled)
thr_tr, _ = find_optimal_threshold_ks(y_dc_tr, p_tr)
m = evaluate("DANN Case 3 (Full DANN)", y_dc_te, p_te, threshold=thr_tr)
dann_ablation.append(m); LOG_AB3.write(fmt_metrics_row(m))
LOG_HO3.write(fmt_metrics_row(m))
LOG_HO3.section("Training history (mỗi 200 step)", level=3)
for h in hist:
    LOG_HO3.write(f"  step {h['step']:4d} | λ={h['lambda']:.4f} | "
                  f"lr={h['lr']:.5f} | L_s={h['L_lab_s']:.4f} | "
                  f"L_d={h['L_dom']:.4f}")


LOG_AB3.section("Bảng tổng hợp 3-case ablation", level=2)
LOG_AB3.write(hdr); LOG_AB3.write("  " + "-" * 70)
for r in dann_ablation:
    LOG_AB3.write(f"  {r['Model']:<25s} {r['AUC']:>8.4f} {r['KS']:>8.4f} "
                  f"{r['Brier']:>8.4f} {r['Sens']:>8.4f} {r['Spec']:>8.4f}")
LOG_AB3.close(); LOG_HO3.close()
LOG3.write(f"Chi tiết hold-out → {OUTPUT_DIR}/03b_dann_holdout.txt")
LOG3.write(f"Chi tiết ablation → {OUTPUT_DIR}/03c_dann_ablation.txt")

# ──────────────────────────────────────────────────────────────────────────────
# 3.7. Tóm tắt phần DANN
# ──────────────────────────────────────────────────────────────────────────────
LOG3.section("3.7. Tóm tắt phần DANN", level=2)
LOG3.write(f"5-fold CV: AUC={dann_auc_mean:.4f}±{dann_auc_std:.4f}, "
           f"KS={dann_ks_mean:.4f}±{dann_ks_std:.4f}")
LOG3.write("Hold-out 3-case:")
for r in dann_ablation:
    LOG3.write(f"  {r['Model']:<25s} | {fmt_metrics_row(r)}")
dann_holdout_main = dann_ablation[2]
LOG3.close()
RUN_LOG.write(f"[Done] PHẦN 3 — DANN → {OUTPUT_DIR}/03_dann.txt")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   FINAL — SO SÁNH BA PHƯƠNG PHÁP & KẾT LUẬN                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
LOG_F = TeeLogger(OUTPUT_DIR / "99_final_comparison.txt")
LOG_F.section("FINAL — SO SÁNH BASELINE / HFA / DANN (DCCC hold-out)", level=1)

# Lấy XGBoost làm đại diện baseline mạnh nhất (theo Lessmann 2015)
best_baseline = max(baseline_holdout_results, key=lambda r: r["AUC"])
LOG_F.write(f"Baseline tốt nhất (theo AUC): {best_baseline['Model']}")
LOG_F.write("")

LOG_F.section("Bảng so sánh tổng (Hold-out)", level=2)
hdr = (f"  {'Method':<28s} {'AUC':>8s} {'KS':>8s} "
       f"{'Brier':>8s} {'Sens':>8s} {'Spec':>8s} {'Acc':>8s}")
LOG_F.write(hdr); LOG_F.write("  " + "-" * 78)
def _row(r):
    LOG_F.write(f"  {r['Model']:<28s} {r['AUC']:>8.4f} {r['KS']:>8.4f} "
                f"{r['Brier']:>8.4f} {r['Sens']:>8.4f} {r['Spec']:>8.4f} "
                f"{r['Acc']:>8.4f}")
for r in baseline_holdout_results:
    _row(r)
LOG_F.write("  " + "-" * 78)
_row(hfa_holdout_main)
_row(dann_holdout_main)

LOG_F.section("Phân tích Negative Transfer", level=2)
b_auc = best_baseline["AUC"]
hfa_auc = hfa_holdout_main["AUC"]
dann_auc = dann_holdout_main["AUC"]
LOG_F.write(f"Baseline tốt nhất AUC = {b_auc:.4f}")
LOG_F.write(f"HFA  full method AUC  = {hfa_auc:.4f}  "
            f"({'+' if hfa_auc>=b_auc else ''}{(hfa_auc-b_auc)*100:.2f} điểm AUC)")
LOG_F.write(f"DANN full method AUC  = {dann_auc:.4f}  "
            f"({'+' if dann_auc>=b_auc else ''}{(dann_auc-b_auc)*100:.2f} điểm AUC)")
LOG_F.write("")
if hfa_auc < b_auc:
    LOG_F.write("⚠️  HFA dưới baseline → có dấu hiệu Negative Transfer.")
else:
    LOG_F.write("✅ HFA vượt baseline → transfer có ích.")
if dann_auc < b_auc:
    LOG_F.write("⚠️  DANN dưới baseline → có dấu hiệu Negative Transfer.")
else:
    LOG_F.write("✅ DANN vượt baseline → transfer có ích.")

LOG_F.section("Kết luận", level=2)
LOG_F.write(
    "Tài liệu này thực thi đầy đủ 3 nhánh phân tích trên cùng cặp dữ liệu "
    "(LC source ↔ DCCC target). Mỗi nhánh được đánh giá qua 5-fold CV và "
    "hold-out, với 3-case ablation kiểm tra Negative Transfer. Tham chiếu "
    "lý thuyết được trích dẫn trực tiếp trong code (Yang 2020 §6.3 cho HFA, "
    "Ganin 2016 §4 cho DANN, Ben-David 2010 cho H-divergence bound).")
LOG_F.close()
RUN_LOG.write(f"[Done] FINAL → {OUTPUT_DIR}/99_final_comparison.txt")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              KẾT THÚC RUN                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
RUN_LOG.write(f"\n[{datetime.datetime.now()}] Run finished.")
RUN_LOG.write("Tóm tắt các file output:")
for fn in sorted(OUTPUT_DIR.glob("*.txt")):
    RUN_LOG.write(f"  • {fn.name}  ({fn.stat().st_size} bytes)")
RUN_LOG.close()
print("\n✅ DONE. Mở thư mục output/ để xem chi tiết.")
