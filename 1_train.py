"""
AgriAI — File 1 of 4: Training Pipeline (Improved)
====================================================
Dataset : Top Agriculture Crop Disease India (Kaggle CC0)
          kaggle.com/datasets/kamaljit_singh/top-agriculture-crop-disease-india

Run:
    pip install -r requirements.txt
    python 1_train.py

Outputs:
    agriai_model.pkl        ← XGBoost classifier (used by server)
    agriai_labels.json      ← Tamil label + action map (used by server)
    agriai_embeddings.csv   ← full-dim PCA-reduced rows (kept for retraining)
    agriai_pca.pkl          ← fitted PCA transformer (used by server)

Improvements over v1:
    - Uses PCA on full 1024-dim MobileNetV3 output (was truncating to 32)
    - Smart augmentation for ALL classes with < 500 images
    - 5-fold stratified cross-validation
    - Progress bars with tqdm
    - Timing per step
"""

import json, pickle, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import xgboost as xgb
import tensorflow as tf

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
DATASET_DIR    = Path("../Crop Diseases")         # unzipped Kaggle folder
TFLITE_MODEL   = Path("mobilenet_v3_small.tflite")
EMBEDDINGS_CSV = Path("agriai_embeddings.csv")
MODEL_OUT      = Path("agriai_model.pkl")
PCA_OUT        = Path("agriai_pca.pkl")
LABELS_OUT     = Path("agriai_labels.json")
IMG_SIZE       = (224, 224)
FULL_EMBED_DIM = 1024                    # MobileNetV3 Small full output
PCA_DIM        = 64                      # reduced via PCA (was hard-truncated to 32)
AUG_THRESHOLD  = 500                     # augment any class below this count
MIN_TARGET     = 500                     # augment to at least this many images

# ── 17 CLASSES: folder name → Tamil name + action ────────────────────────────
CLASSES = {
    "Rice___Brown_Spot":            ("நெல் - பழுப்பு புள்ளி",       "கார்பெண்டசிம் தெளிக்கவும். நீர் தேக்கம் தவிர்க்கவும்."),
    "Rice___Healthy":               ("நெல் - ஆரோக்கியமான",           "நோய் இல்லை. தொடர்ந்து கவனிக்கவும்."),
    "Rice___Leaf_Blast":            ("நெல் - இலை வெடிப்பு",          "ட்ரைசைக்லோசோல் தெளிக்கவும். 7 நாள் இடைவெளி."),
    "Rice___Neck_Blast":            ("நெல் - கழுத்து வெடிப்பு",      "உடனடியாக ட்ரைசைக்லோசோல் தெளிக்கவும்."),
    "Wheat___Brown_Rust":           ("கோதுமை - பழுப்பு துரு",        "புரோபிக்கோனசோல் தெளிக்கவும். 14 நாள் இடைவெளி."),
    "Wheat___Healthy":              ("கோதுமை - ஆரோக்கியமான",          "நோய் இல்லை. தொடர்ந்து கவனிக்கவும்."),
    "Wheat___Yellow_Rust":          ("கோதுமை - மஞ்சள் துரு",         "டெபுக்கோனசோல் தெளிக்கவும். உடனடியாக செயல்படவும்."),
    "Sugarcane_Bacterial Blight":   ("கரும்பு - பாக்டீரியா நோய்",    "நோய் தாக்கிய பகுதி அகற்றவும். காப்பர் கலவை தெளிக்கவும்."),
    "Sugarcane_Healthy":            ("கரும்பு - ஆரோக்கியமான",         "நோய் இல்லை. தொடர்ந்து கவனிக்கவும்."),
    "Sugarcane_Red Rot":            ("கரும்பு - சிவப்பு அழுகல்",     "நோய் தாக்கிய கரும்பை அகற்றவும். விவசாயியை அழைக்கவும்."),
    "Corn___Common_Rust":           ("மக்காச்சோளம் - பொது துரு",      "மேன்கோசெப் தெளிக்கவும். 10 நாள் இடைவெளி."),
    "Corn___Gray_Leaf_Spot":        ("மக்காச்சோளம் - சாம்பல் புள்ளி","அசோக்ஸிஸ்ட்ரோபின் தெளிக்கவும்."),
    "Corn___Healthy":               ("மக்காச்சோளம் - ஆரோக்கியமான",   "நோய் இல்லை. தொடர்ந்து கவனிக்கவும்."),
    "Corn___Northern_Leaf_Blight":  ("மக்காச்சோளம் - வடக்கு கருகல்", "புரோபிக்கோனசோல் தெளிக்கவும். மழைக்கு பின் மீண்டும் தெளிக்கவும்."),
    "Potato___Early_Blight":        ("உருளை - ஆரம்ப கருகல்",         "குளோரோத்தாலோனில் தெளிக்கவும். 7 நாள் இடைவெளி."),
    "Potato___Healthy":             ("உருளை - ஆரோக்கியமான",           "நோய் இல்லை. தொடர்ந்து கவனிக்கவும்."),
    "Potato___Late_Blight":         ("உருளை - பிற்கால கருகல்",        "மேட்டாலக்ஸில் தெளிக்கவும். உடனடியாக செயல்படவும்."),
}

LABEL_TO_ID = {cls: i for i, cls in enumerate(sorted(CLASSES))}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


# ── AUGMENTATION (for any class with < AUG_THRESHOLD images) ──────────────────
def augment(img: Image.Image) -> list:
    """Generate 5 augmented variants of an image."""
    return [
        ImageOps.mirror(img),
        img.rotate(15),
        img.rotate(-15),
        ImageEnhance.Brightness(img).enhance(1.4),
        ImageEnhance.Brightness(img).enhance(0.65),
    ]


# ── PREPROCESSING (must match Android app byte-for-byte) ─────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    for c in range(3):                              # CLAHE approximation
        ch = arr[:, :, c]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            arr[:, :, c] = np.clip((ch - p2) / (p98 - p2) * 255, 0, 255)
    return (arr / 255.0).astype(np.float32)


# ── TFLITE EMBEDDING ──────────────────────────────────────────────────────────
def load_tflite(path: Path):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp

def embed(interp, arr: np.ndarray) -> np.ndarray:
    """Extract FULL 1024-dim embedding from MobileNetV3."""
    inp = interp.get_input_details()
    out = interp.get_output_details()
    interp.set_tensor(inp[0]["index"], arr[np.newaxis])
    interp.invoke()
    return interp.get_tensor(out[0]["index"])[0]    # full 1024 dims


# ── STEP 1: BUILD EMBEDDINGS ──────────────────────────────────────────────────
def build_embeddings():
    if not TFLITE_MODEL.exists():
        print("ERROR: mobilenet_v3_small.tflite not found.")
        print("Get it with:")
        print("  wget https://storage.googleapis.com/tfhub-lite-models/google/"
              "lite-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1.tflite"
              " -O mobilenet_v3_small.tflite")
        return None

    t0 = time.time()
    interp = load_tflite(TFLITE_MODEL)
    rows, errors, total = [], 0, 0

    for cls_name, (tamil, action) in CLASSES.items():
        cls_dir = DATASET_DIR / cls_name
        if not cls_dir.exists():
            print(f"  MISSING: {cls_dir}"); continue

        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG")) + \
               list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        lid  = LABEL_TO_ID[cls_name]

        # Smart augmentation: augment ANY class below AUG_THRESHOLD
        do_aug = len(imgs) < AUG_THRESHOLD
        tag = f"[AUG {len(imgs)}→~{len(imgs)*6}]" if do_aug else ""
        print(f"  {cls_name}: {len(imgs)} imgs {tag}")

        for p in tqdm(imgs, desc=f"  {cls_name[:25]}", leave=False):
            try:
                img      = Image.open(p)
                variants = [img] + (augment(img) if do_aug else [])
                for v in variants:
                    emb = embed(interp, preprocess(v))
                    rows.append(list(emb) + [lid, cls_name])
                    total += 1
            except Exception as e:
                errors += 1

    feat = [f"f{i}" for i in range(FULL_EMBED_DIM)]
    df   = pd.DataFrame(rows, columns=feat + ["label_id", "class_name"])
    df.to_csv(EMBEDDINGS_CSV, index=False)
    elapsed = time.time() - t0
    print(f"\nSaved {total} embeddings → {EMBEDDINGS_CSV}  (errors: {errors}, time: {elapsed:.0f}s)")
    print(df["class_name"].value_counts().to_string())
    return df


# ── STEP 2: PCA REDUCTION ────────────────────────────────────────────────────
def fit_pca(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce 1024-dim embeddings to PCA_DIM dimensions."""
    feat = [f"f{i}" for i in range(FULL_EMBED_DIM)]
    X    = df[feat].values

    print(f"\n  Fitting PCA: {FULL_EMBED_DIM}→{PCA_DIM} dims on {len(X)} samples...")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X_pca = pca.fit_transform(X)

    var_explained = sum(pca.explained_variance_ratio_) * 100
    print(f"  PCA variance explained: {var_explained:.1f}%")

    with open(PCA_OUT, "wb") as f:
        pickle.dump(pca, f)
    print(f"  PCA transformer → {PCA_OUT}")

    # Build reduced dataframe
    pca_feat = [f"p{i}" for i in range(PCA_DIM)]
    df_pca = pd.DataFrame(X_pca, columns=pca_feat)
    df_pca["label_id"]    = df["label_id"].values
    df_pca["class_name"]  = df["class_name"].values
    return df_pca


# ── STEP 3: TRAIN XGBOOST WITH CROSS-VALIDATION ──────────────────────────────
def train_model():
    df   = pd.read_csv(EMBEDDINGS_CSV)
    df_pca = fit_pca(df)

    pca_feat = [f"p{i}" for i in range(PCA_DIM)]
    X, y = df_pca[pca_feat].values, df_pca["label_id"].values

    # ── 5-Fold Stratified Cross-Validation ────────────────────────────────────
    print(f"\n  5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        clf_cv = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
            verbosity=0,
        )
        clf_cv.fit(X[train_idx], y[train_idx],
                   eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        fold_acc = accuracy_score(y[val_idx], clf_cv.predict(X[val_idx]))
        cv_scores.append(fold_acc)
        print(f"    Fold {fold}: {fold_acc*100:.1f}%")

    mean_cv = np.mean(cv_scores)
    std_cv  = np.std(cv_scores)
    print(f"  CV Accuracy: {mean_cv*100:.1f}% ± {std_cv*100:.1f}%")

    # ── Final model on 80/20 split ────────────────────────────────────────────
    print(f"\n  Training final model on 80/20 split...")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"  Train: {len(Xtr)}  Test: {len(Xte)}")

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=50)

    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"\n  Final Accuracy: {acc*100:.1f}%")
    print(classification_report(yte, clf.predict(Xte),
          target_names=[ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]))

    with open(MODEL_OUT, "wb") as f: pickle.dump(clf, f)
    print(f"  Model → {MODEL_OUT}")
    return acc, mean_cv


# ── STEP 4: SAVE LABEL MAP ───────────────────────────────────────────────────
def save_labels():
    out = {
        "label_to_id": LABEL_TO_ID,
        "pca_dim": PCA_DIM,
        "full_embed_dim": FULL_EMBED_DIM,
        "id_to_info": {
            str(LABEL_TO_ID[cls]): {
                "class": cls,
                "tamil": tamil,
                "action": action,
            }
            for cls, (tamil, action) in CLASSES.items()
        },
    }
    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  Labels → {LABELS_OUT}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("AgriAI Training Pipeline v2 (Improved)")
    print(f"Classes: {len(CLASSES)}  |  Dataset: {DATASET_DIR}")
    print(f"Embedding: {FULL_EMBED_DIM}→PCA({PCA_DIM})  |  Aug threshold: <{AUG_THRESHOLD}")
    print("=" * 60)

    save_labels()

    if not EMBEDDINGS_CSV.exists():
        print("\n[1/3] Extracting TFLite embeddings (full 1024-dim)...")
        build_embeddings()
    else:
        print(f"\n[1/3] Found {EMBEDDINGS_CSV}, skipping extraction.")

    print("\n[2/3] PCA reduction + XGBoost training with 5-fold CV...")
    acc, cv_acc = train_model()

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Done!  Final Accuracy: {acc*100:.1f}%  |  CV: {cv_acc*100:.1f}%")
    print(f"Time: {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"Next:  run  2_server.py")
    print("=" * 60)
