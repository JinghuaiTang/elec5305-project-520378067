# ================================================================
# Evaluating Classical Spectral Features and Shallow Classifiers
# for Urban Sound Classification
# ================================================================

# ------------------------------------------------
# Module 0: Imports and Global Configuration
# ------------------------------------------------
import os
import sys
import time
import math
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

try:
    import librosa
except ImportError as e:
    raise ImportError("librosa is required. Please install it with 'pip install librosa'.") from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("matplotlib is required. Please install it with 'pip install matplotlib'.") from e

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
        precision_score,
        recall_score,
    )
except ImportError as e:
    raise ImportError("scikit-learn is required. Please install it with 'pip install scikit-learn'.") from e


# ------------------------------------------------
# Module 1: Utility Functions and Experiment Config
# ------------------------------------------------
@dataclass
class ExperimentConfig:
    """Configuration object for the classical-feature urban sound experiment."""
    # Audio and feature parameters
    sample_rate: int = 22050
    target_duration: float = 4.0  # seconds
    n_mfcc: int = 20
    use_delta: bool = True
    use_delta_delta: bool = False
    include_centroid: bool = True
    include_bandwidth: bool = True
    include_rolloff: bool = True
    include_flatness: bool = True
    include_zcr: bool = True

    # Data selection
    folds_to_use: List[int] = None           # e.g. [1, 2, 3]
    classes_to_use: Optional[List[str]] = None  # e.g. ["car_horn", "dog_bark", "street_music"]
    max_files_per_class: Optional[int] = 120   # to keep runtime moderate

    # Train/validation/test split
    test_size: float = 0.2
    val_size: float = 0.2   # fraction of train_val that becomes validation
    random_seed: int = 42

    # Cross-validation for model comparison
    use_cross_val: bool = True
    cv_folds: int = 5

    # Classifiers to evaluate
    evaluate_svm: bool = True
    evaluate_logreg: bool = True
    evaluate_rf: bool = True

    # Paths and output
    results_dir_name: str = "results_classical"
    run_name: str = "urban_sounds_classical_features"

    # Feature caching
    use_feature_cache: bool = True
    feature_cache_dir_name: str = "feature_cache"

    def __post_init__(self):
        if self.folds_to_use is None:
            # Default to a small subset of folds for faster experiments
            self.folds_to_use = [1, 2, 3]


def get_project_root() -> Path:
    """Infer the project root as the directory where this script lives."""
    try:
        root = Path(__file__).resolve().parent
    except NameError:
        root = Path(".").resolve()
    return root


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def format_seconds(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# ------------------------------------------------
# Module 2: Metadata Loading and Subsetting (UrbanSound8K style)
# ------------------------------------------------
def locate_metadata_csv(dataset_root: Path) -> Path:
    """
    Try to locate the UrbanSound8K metadata CSV.

    Expected locations:
      - Dataset/metadata/UrbanSound8K.csv
      - Dataset/UrbanSound8K.csv
    """
    candidate1 = dataset_root / "metadata" / "UrbanSound8K.csv"
    candidate2 = dataset_root / "UrbanSound8K.csv"
    if candidate1.is_file():
        return candidate1
    if candidate2.is_file():
        return candidate2
    raise FileNotFoundError(
        f"Could not find UrbanSound8K metadata CSV. "
        f"Tried {candidate1} and {candidate2}."
    )


def load_metadata(dataset_root: Path) -> pd.DataFrame:
    """Load UrbanSound8K metadata as a DataFrame."""
    csv_path = locate_metadata_csv(dataset_root)
    df = pd.read_csv(csv_path)
    required_cols = {"slice_file_name", "fold", "classID", "class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {missing}")
    return df


def subset_metadata(
    df: pd.DataFrame,
    folds_to_use: List[int],
    classes_to_use: Optional[List[str]] = None,
    max_files_per_class: Optional[int] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Subset metadata by folds and optionally by class name.
    If max_files_per_class is not None, it applies to each (class, fold) pair
    instead of the whole class across all folds.
    """
    # Filter by folds
    df_sub = df[df["fold"].isin(folds_to_use)].copy()

    # Filter by class if specified
    if classes_to_use is not None:
        df_sub = df_sub[df_sub["class"].isin(classes_to_use)].copy()

    # Limit samples per (class, fold) if requested
    if max_files_per_class is not None:
        grouped_chunks = []
        rng = np.random.default_rng(random_seed)

        for (cls_name, fold), group in df_sub.groupby(["class", "fold"]):
            if len(group) > max_files_per_class:
                indices = rng.choice(group.index, size=max_files_per_class, replace=False)
                grouped_chunks.append(group.loc[indices])
            else:
                grouped_chunks.append(group)

        df_sub = pd.concat(grouped_chunks, ignore_index=True)

    df_sub = df_sub.reset_index(drop=True)

    # Print basic stats per class
    print("Samples per class after subsetting:")
    for cls_name, g in df_sub.groupby("class"):
        folds_present = sorted(g["fold"].unique().tolist())
        print(f"  {cls_name:20s}: total {len(g):4d}  | folds: {folds_present}")

    return df_sub


def build_label_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """
    Build a mapping from class name to integer label index.
    """
    classes = sorted(df["class"].unique().tolist())
    label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return label_map


# ------------------------------------------------
# Module 3: Audio Loading and Classical Feature Extraction
# ------------------------------------------------
def load_waveform(
    file_path: Path,
    sample_rate: int,
    target_duration: float,
) -> np.ndarray:
    """
    Load a waveform with librosa, resample to sample_rate and pad or trim to target_duration.
    """
    y, sr = librosa.load(str(file_path), sr=sample_rate, mono=True)
    target_length = int(sample_rate * target_duration)
    if len(y) < target_length:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode="constant")
    else:
        y = y[:target_length]
    return y.astype(np.float32)


def compute_mfcc_features(
    y: np.ndarray,
    sample_rate: int,
    n_mfcc: int,
    use_delta: bool = True,
    use_delta_delta: bool = False,
) -> np.ndarray:
    """
    Compute MFCC features and aggregate them into a fixed-length vector.

    For each MFCC coefficient, we compute statistical summaries such as mean and std.
    Optionally we append delta and delta-delta MFCCs.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    stats = [mfcc.mean(axis=1), mfcc.std(axis=1)]
    if use_delta:
        delta1 = librosa.feature.delta(mfcc)
        stats.append(delta1.mean(axis=1))
        stats.append(delta1.std(axis=1))
    if use_delta_delta:
        delta2 = librosa.feature.delta(mfcc, order=2)
        stats.append(delta2.mean(axis=1))
        stats.append(delta2.std(axis=1))
    feature_vec = np.concatenate(stats, axis=0)
    return feature_vec.astype(np.float32)


def compute_spectral_features(
    y: np.ndarray,
    sample_rate: int,
    include_centroid: bool = True,
    include_bandwidth: bool = True,
    include_rolloff: bool = True,
    include_flatness: bool = True,
    include_zcr: bool = True,
) -> np.ndarray:
    """
    Compute various simple spectral features and aggregate them as statistics.
    """
    features = []

    # Spectral centroid
    if include_centroid:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
        features.append(centroid.mean(axis=1))
        features.append(centroid.std(axis=1))

    # Spectral bandwidth
    if include_bandwidth:
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)
        features.append(bandwidth.mean(axis=1))
        features.append(bandwidth.std(axis=1))

    # Spectral rolloff
    if include_rolloff:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate, roll_percent=0.85)
        features.append(rolloff.mean(axis=1))
        features.append(rolloff.std(axis=1))

    # Spectral flatness
    if include_flatness:
        flatness = librosa.feature.spectral_flatness(y=y)
        features.append(flatness.mean(axis=1))
        features.append(flatness.std(axis=1))

    # Zero crossing rate
    if include_zcr:
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(zcr.mean(axis=1))
        features.append(zcr.std(axis=1))

    if len(features) == 0:
        return np.zeros((0,), dtype=np.float32)

    feature_vec = np.concatenate(features, axis=0)
    return feature_vec.astype(np.float32)


def get_feature_cache_path(
    cache_root: Optional[Path],
    fold: int,
    slice_file_name: str,
) -> Optional[Path]:
    """
    Compute a cache path for a given file if cache_root is provided.
    """
    if cache_root is None:
        return None
    stem = Path(slice_file_name).stem
    fold_dir = cache_root / f"fold{fold}"
    ensure_dir(fold_dir)
    return fold_dir / f"{stem}.npy"


def extract_features_for_file(
    file_path: Path,
    sample_rate: int,
    target_duration: float,
    n_mfcc: int,
    use_delta: bool,
    use_delta_delta: bool,
    include_centroid: bool,
    include_bandwidth: bool,
    include_rolloff: bool,
    include_flatness: bool,
    include_zcr: bool,
) -> np.ndarray:
    """
    Extract combined MFCC + spectral features for a single audio file.
    """
    y = load_waveform(file_path, sample_rate, target_duration)
    mfcc_feat = compute_mfcc_features(
        y,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        use_delta=use_delta,
        use_delta_delta=use_delta_delta,
    )
    spec_feat = compute_spectral_features(
        y,
        sample_rate=sample_rate,
        include_centroid=include_centroid,
        include_bandwidth=include_bandwidth,
        include_rolloff=include_rolloff,
        include_flatness=include_flatness,
        include_zcr=include_zcr,
    )
    combined = np.concatenate([mfcc_feat, spec_feat], axis=0)
    return combined.astype(np.float32)


def load_or_compute_features(
    cache_path: Optional[Path],
    file_path: Path,
    config: ExperimentConfig,
) -> np.ndarray:
    """
    Load features from disk if cache exists, otherwise compute and optionally save.
    """
    if cache_path is not None and cache_path.is_file():
        arr = np.load(cache_path)
        return arr.astype(np.float32)

    features = extract_features_for_file(
        file_path=file_path,
        sample_rate=config.sample_rate,
        target_duration=config.target_duration,
        n_mfcc=config.n_mfcc,
        use_delta=config.use_delta,
        use_delta_delta=config.use_delta_delta,
        include_centroid=config.include_centroid,
        include_bandwidth=config.include_bandwidth,
        include_rolloff=config.include_rolloff,
        include_flatness=config.include_flatness,
        include_zcr=config.include_zcr,
    )

    if cache_path is not None:
        np.save(cache_path, features)

    return features


# ------------------------------------------------
# Module 4: Feature Matrix Construction
# ------------------------------------------------
def build_feature_matrix(
    df: pd.DataFrame,
    dataset_root: Path,
    label_map: Dict[str, int],
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build the feature matrix X and label vector y for a given subset of metadata.

    Returns:
      X: shape (N_samples, D_features)
      y: shape (N_samples,)
      file_ids: list of identifiers (e.g. file names) for bookkeeping
    """
    audio_root = dataset_root / "audio"
    if not audio_root.is_dir():
        raise FileNotFoundError(
            f"Expected audio root at {audio_root}, with subfolders fold1..fold10."
        )

    cache_root = None
    if config.use_feature_cache:
        cache_root = dataset_root / config.feature_cache_dir_name
        ensure_dir(cache_root)

    feature_list: List[np.ndarray] = []
    label_list: List[int] = []
    file_ids: List[str] = []

    total = len(df)
    t0 = time.perf_counter()
    print(f"Starting feature extraction for {total} files ...")

    for idx, row in df.iterrows():
        slice_file_name = row["slice_file_name"]
        fold = int(row["fold"])
        cls_name = str(row["class"])

        audio_path = audio_root / f"fold{fold}" / slice_file_name
        if not audio_path.is_file():
            print(f"[Warning] Audio file not found: {audio_path}, skipping.")
            continue

        cache_path = get_feature_cache_path(cache_root, fold, slice_file_name)

        feat = load_or_compute_features(
            cache_path=cache_path,
            file_path=audio_path,
            config=config,
        )

        label_idx = label_map[cls_name]

        feature_list.append(feat)
        label_list.append(label_idx)
        file_ids.append(slice_file_name)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            elapsed = format_seconds(time.perf_counter() - t0)
            print(f"  Processed {idx+1}/{total} files ... ({elapsed})")

    if not feature_list:
        raise RuntimeError("No features were extracted. Check dataset paths and filters.")

    X = np.stack(feature_list, axis=0)
    y = np.array(label_list, dtype=np.int64)
    return X, y, file_ids


# ------------------------------------------------
# Module 5: Dataset Splitting (Train/Val/Test)
# ------------------------------------------------
def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified split into train, validation and test sets.

    Steps:
      1) Split into train_val and test.
      2) Further split train_val into train and val.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    val_fraction_of_train_val = val_size
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_train_val,
        random_state=random_seed,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------
# Module 6: Classifier Definitions and Cross-Validation
# ------------------------------------------------
def build_classifiers(random_seed: int = 42) -> Dict[str, Pipeline]:
    """
    Build a dictionary of shallow classifier pipelines using scikit-learn.

    Each classifier is wrapped in a Pipeline with a StandardScaler (except RF).
    """
    classifiers: Dict[str, Pipeline] = {}

    # Support Vector Machine with RBF kernel
    svm_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=random_seed)),
        ]
    )
    classifiers["svm_rbf"] = svm_pipeline

    # Logistic Regression (multinomial, L2)
    logreg_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                multi_class="multinomial",
                max_iter=200,
                random_state=random_seed,
            )),
        ]
    )
    classifiers["logreg"] = logreg_pipeline

    # Random Forest (no scaler needed)
    rf_pipeline = Pipeline(
        steps=[
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=random_seed,
                n_jobs=-1,
            )),
        ]
    )
    classifiers["random_forest"] = rf_pipeline

    return classifiers


def cross_validate_classifier(
    name: str,
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform cross-validation for a given classifier pipeline and return summary metrics.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    acc_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
    f1_macro_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1_macro")

    result = {
        "classifier": name,
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std": float(acc_scores.std()),
        "cv_macro_f1_mean": float(f1_macro_scores.mean()),
        "cv_macro_f1_std": float(f1_macro_scores.std()),
    }
    return result


# ------------------------------------------------
# Module 7: Training, Evaluation and Reporting on Train/Val/Test
# ------------------------------------------------
def train_and_evaluate_classifier(
    name: str,
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Train a classifier on train + val data, evaluate on val and test, and collect metrics.
    """
    # Fit on combined train + val
    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.concatenate([y_train, y_val])
    print(f"\n[{name}] Fitting classifier on {len(y_train_combined)} samples ...")
    t0 = time.perf_counter()
    pipeline.fit(X_train_combined, y_train_combined)
    elapsed_train = format_seconds(time.perf_counter() - t0)
    print(f"[{name}] Training completed in {elapsed_train}.")

    # Evaluate on validation set
    print(f"[{name}] Evaluating on validation set ({len(y_val)} samples) ...")
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    # Evaluate on test set
    print(f"[{name}] Evaluating on test set ({len(y_test)} samples) ...")
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")

    # Classification report and confusion matrix for test set
    report = classification_report(
        y_test,
        y_test_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_test_pred)

    print(f"\n[{name}] Validation accuracy: {val_acc:.4f}, macro-F1: {val_macro_f1:.4f}")
    print(f"[{name}] Test accuracy: {test_acc:.4f}, macro-F1: {test_macro_f1:.4f}")
    print(f"\n[{name}] Test classification report:\n{report}")

    result = {
        "classifier": name,
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    return result


# ------------------------------------------------
# Module 8: Visualization Helpers
# ------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
) -> None:
    """
    Plot and save a confusion matrix as an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_histograms(
    X: np.ndarray,
    feature_names: List[str],
    out_path: Path,
    max_features_to_plot: int = 20,
) -> None:
    """
    Plot histograms for the first few features to get a sense of their distributions.
    """
    num_features = min(len(feature_names), max_features_to_plot)
    cols = 4
    rows = math.ceil(num_features / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten()

    for i in range(num_features):
        ax = axes[i]
        ax.hist(X[:, i], bins=30, alpha=0.7)
        ax.set_title(feature_names[i], fontsize=8)
    for j in range(num_features, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------
# Module 9: High-Level Orchestration Function
# ------------------------------------------------
def run_experiment(config: ExperimentConfig) -> None:
    """
    Orchestrate the full pipeline:
      1) Load and subset metadata
      2) Build feature matrix
      3) Split into train/val/test
      4) Optional cross-validation on full data
      5) Train and evaluate each classifier
      6) Save results and plots
      7) Run additional analysis modules
    """
    seed_everything(config.random_seed)

    project_root = get_project_root()
    dataset_root = project_root / "Dataset"
    audio_root = dataset_root / "audio"
    if not audio_root.is_dir():
        raise FileNotFoundError(
            f"Audio root folder not found at {audio_root}. "
            f"Expected structure: (script folder)/Dataset/audio/fold1..fold10"
        )

    results_root = project_root / config.results_dir_name / config.run_name
    ensure_dir(results_root)

    # 1) Load and subset metadata
    print("Loading metadata ...")
    df_meta = load_metadata(dataset_root)
    df_sub = subset_metadata(
        df_meta,
        folds_to_use=config.folds_to_use,
        classes_to_use=config.classes_to_use,
        max_files_per_class=config.max_files_per_class,
        random_seed=config.random_seed,
    )
    print(f"Subset metadata: {len(df_sub)} rows "
          f"(folds={config.folds_to_use}, "
          f"classes={config.classes_to_use}, "
          f"max_files_per_class={config.max_files_per_class})")

    if df_sub["class"].nunique() < 2:
        raise ValueError(
            "Need at least 2 distinct classes in the subset to perform classification."
        )

    # Build label mapping and feature names
    label_map = build_label_mapping(df_sub)
    inverse_label_map = {v: k for k, v in label_map.items()}
    class_names = [inverse_label_map[i] for i in range(len(inverse_label_map))]

    # 2) Build feature matrix
    X, y, file_ids = build_feature_matrix(df_sub, dataset_root, label_map, config)
    print(f"Feature matrix shape: X={X.shape}, y={y.shape}")

    # Build simple feature names for interpretability
    feature_names: List[str] = []
    stat_names = ["mean", "std"]
    if config.use_delta:
        stat_names.extend(["delta_mean", "delta_std"])
    if config.use_delta_delta:
        stat_names.extend(["delta2_mean", "delta2_std"])
    for stat in stat_names:
        for i in range(config.n_mfcc):
            feature_names.append(f"mfcc_{i+1}_{stat}")

    def add_spec_names(prefix: str):
        feature_names.append(f"{prefix}_mean")
        feature_names.append(f"{prefix}_std")

    if config.include_centroid:
        add_spec_names("centroid")
    if config.include_bandwidth:
        add_spec_names("bandwidth")
    if config.include_rolloff:
        add_spec_names("rolloff")
    if config.include_flatness:
        add_spec_names("flatness")
    if config.include_zcr:
        add_spec_names("zcr")

    # 3) Split into train / val / test
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X,
        y,
        test_size=config.test_size,
        val_size=config.val_size,
        random_seed=config.random_seed,
    )
    print(f"Split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # 4) Dataset-level summary and plots (new Module 10)
    summarize_dataset_and_plot(df_sub, class_names, results_root)

    # 5) Evaluate a simple majority-class baseline (new Module 11)
    evaluate_majority_class_baseline(y_train, y_val, y_test, class_names, results_root)

    # 6) Optional cross-validation (on full X, y)
    cv_results: List[Dict[str, Any]] = []
    if config.use_cross_val:
        print("\nRunning cross-validation on full dataset (X, y) ...")
        classifiers = build_classifiers(config.random_seed)
        for clf_name, pipeline in classifiers.items():
            print(f"  Cross-validating classifier: {clf_name}")
            res = cross_validate_classifier(
                name=clf_name,
                pipeline=pipeline,
                X=X,
                y=y,
                cv_folds=config.cv_folds,
                random_seed=config.random_seed,
            )
            cv_results.append(res)
            print(
                f"    CV accuracy: {res['cv_accuracy_mean']:.4f} ± {res['cv_accuracy_std']:.4f}, "
                f"macro-F1: {res['cv_macro_f1_mean']:.4f} ± {res['cv_macro_f1_std']:.4f}"
            )

        df_cv = pd.DataFrame(cv_results)
        df_cv.to_csv(results_root / "cross_validation_summary.csv", index=False)
    else:
        classifiers = build_classifiers(config.random_seed)

    # 7) Train and evaluate chosen classifiers on train/val/test
    train_test_results: List[Dict[str, Any]] = []
    for clf_name, pipeline in classifiers.items():
        if clf_name == "svm_rbf" and not config.evaluate_svm:
            continue
        if clf_name == "logreg" and not config.evaluate_logreg:
            continue
        if clf_name == "random_forest" and not config.evaluate_rf:
            continue

        result = train_and_evaluate_classifier(
            name=clf_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
        )
        train_test_results.append(result)

        cm = np.array(result["confusion_matrix"])
        cm_out_path = results_root / f"cm_{clf_name}.png"
        plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            title=f"{clf_name} - Test Confusion Matrix",
            out_path=cm_out_path,
        )

        # 8) Save per-class metrics for each classifier (new Module 12)
        save_per_class_metrics_for_classifier(
            name=clf_name,
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            results_root=results_root,
        )

        # 9) Analyze RF feature importance if available (new Module 13)
        if clf_name == "random_forest":
            analyze_random_forest_feature_importance(
                pipeline=pipeline,
                feature_names=feature_names,
                results_root=results_root,
                top_k=20,
            )

    # 10) Save train/val/test summary
    if train_test_results:
        rows = []
        for res in train_test_results:
            rows.append(
                {
                    "classifier": res["classifier"],
                    "val_accuracy": res["val_accuracy"],
                    "val_macro_f1": res["val_macro_f1"],
                    "test_accuracy": res["test_accuracy"],
                    "test_macro_f1": res["test_macro_f1"],
                }
            )
        df_summary = pd.DataFrame(rows)
        df_summary.to_csv(results_root / "train_val_test_summary.csv", index=False)
        print("\nTrain/validation/test summary:")
        print(df_summary.to_string(index=False))

    # 11) Optional: Feature histogram plot for sanity check
    hist_out_path = results_root / "feature_histograms.png"
    plot_feature_histograms(
        X,
        feature_names=feature_names,
        out_path=hist_out_path,
        max_features_to_plot=24,
    )

    # Save config and some metadata
    with open(results_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    subset_info = {
        "folds_to_use": config.folds_to_use,
        "classes_to_use": config.classes_to_use,
        "max_files_per_class": config.max_files_per_class,
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "class_names": class_names,
    }
    with open(results_root / "subset_info.json", "w", encoding="utf-8") as f:
        json.dump(subset_info, f, indent=2)

    # 12) Export example spectrograms per class (new Module 14)
    export_example_spectrograms_per_class(
        df_sub=df_sub,
        dataset_root=dataset_root,
        config=config,
        class_names=class_names,
        results_root=results_root,
        max_examples_per_class=1,
    )

    print("\nExperiment completed. Results saved to:")
    print(results_root)


# ------------------------------------------------
# Module 10: Dataset Diagnostics and Class Distribution
# ------------------------------------------------
def summarize_dataset_and_plot(
    df_sub: pd.DataFrame,
    class_names: List[str],
    results_root: Path,
) -> None:
    """
    Summarize dataset statistics and plot class distribution.
    """
    print("\nDataset summary (subset used in this experiment):")
    counts = df_sub["class"].value_counts()
    counts = counts.reindex(class_names, fill_value=0)
    total = int(counts.sum())
    fractions = counts / max(total, 1)

    for cls, cnt in counts.items():
        frac = fractions[cls]
        print(f"  {cls:20s}: {cnt:4d} samples ({frac*100:5.1f}%)")

    df_stats = pd.DataFrame(
        {
            "class": counts.index.tolist(),
            "count": counts.values.astype(int),
            "fraction": fractions.values.astype(float),
        }
    )
    df_stats.to_csv(results_root / "dataset_class_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_stats["class"], df_stats["count"])
    ax.set_title("Class distribution in selected subset")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(results_root / "dataset_class_distribution.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------
# Module 11: Majority-Class Baseline Evaluation
# ------------------------------------------------
def evaluate_majority_class_baseline(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    results_root: Path,
) -> None:
    """
    Evaluate a simple majority-class baseline that always predicts the most frequent class in training data.
    """
    y_train_val = np.concatenate([y_train, y_val])
    counts = np.bincount(y_train_val)
    majority_label = int(np.argmax(counts))
    majority_class_name = class_names[majority_label]

    def eval_on_split(y_true: np.ndarray) -> Tuple[float, float]:
        y_pred = np.full_like(y_true, fill_value=majority_label)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return acc, macro_f1

    train_acc, train_f1 = eval_on_split(y_train)
    val_acc, val_f1 = eval_on_split(y_val)
    test_acc, test_f1 = eval_on_split(y_test)

    print("\nMajority-class baseline (no learning):")
    print(f"  Majority class: {majority_class_name} (label={majority_label})")
    print(f"  Train accuracy: {train_acc:.4f}, macro-F1: {train_f1:.4f}")
    print(f"  Val   accuracy: {val_acc:.4f}, macro-F1: {val_f1:.4f}")
    print(f"  Test  accuracy: {test_acc:.4f}, macro-F1: {test_f1:.4f}")

    df_baseline = pd.DataFrame(
        [
            ["train", train_acc, train_f1],
            ["val", val_acc, val_f1],
            ["test", test_acc, test_f1],
        ],
        columns=["split", "accuracy", "macro_f1"],
    )
    df_baseline.to_csv(results_root / "baseline_majority_class_summary.csv", index=False)


# ------------------------------------------------
# Module 12: Per-Class Metrics for Each Classifier
# ------------------------------------------------
def save_per_class_metrics_for_classifier(
    name: str,
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    results_root: Path,
) -> None:
    """
    Compute and save per-class precision, recall and F1-score for a given classifier.
    """
    y_pred = pipeline.predict(X_test)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    df = pd.DataFrame(
        {
            "class": class_names,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }
    )
    out_path = results_root / f"per_class_metrics_{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[{name}] Per-class metrics saved to {out_path.name}")


# ------------------------------------------------
# Module 13: Random Forest Feature Importance Analysis
# ------------------------------------------------
def analyze_random_forest_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    results_root: Path,
    top_k: int = 20,
) -> None:
    """
    Extract and save feature importance from a Random Forest classifier.
    """
    clf = pipeline.named_steps.get("clf", None)
    if clf is None or not hasattr(clf, "feature_importances_"):
        print("[random_forest] No feature_importances_ attribute found; skipping importance analysis.")
        return

    importances = np.array(clf.feature_importances_, dtype=float)
    if len(importances) != len(feature_names):
        print("[random_forest] Feature importance length does not match number of features; skipping.")
        return

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    imp_path = results_root / "random_forest_feature_importances.csv"
    df_imp.to_csv(imp_path, index=False)
    print(f"[random_forest] Feature importances saved to {imp_path.name}")

    k = min(top_k, len(df_imp))
    df_top = df_imp.head(k)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df_top["feature"][::-1], df_top["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {k} features by Random Forest importance")
    fig.tight_layout()
    fig.savefig(results_root / "random_forest_top_features.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------
# Module 14: Example Spectrogram Export per Class
# ------------------------------------------------
def export_example_spectrograms_per_class(
    df_sub: pd.DataFrame,
    dataset_root: Path,
    config: ExperimentConfig,
    class_names: List[str],
    results_root: Path,
    max_examples_per_class: int = 1,
) -> None:
    """
    Export one example spectrogram (or a few) per class to visualize the audio content.
    """
    audio_root = dataset_root / "audio"
    if not audio_root.is_dir():
        print("Audio root not found while exporting spectrograms; skipping.")
        return

    print("\nExporting example spectrograms per class ...")
    for cls_name in class_names:
        df_cls = df_sub[df_sub["class"] == cls_name]
        if df_cls.empty:
            continue

        df_examples = df_cls.head(max_examples_per_class)
        for idx, row in df_examples.iterrows():
            fold = int(row["fold"])
            slice_file_name = row["slice_file_name"]
            audio_path = audio_root / f"fold{fold}" / slice_file_name

            if not audio_path.is_file():
                print(f"  [Warning] Audio file not found for spectrogram: {audio_path}")
                continue

            y, sr = librosa.load(str(audio_path), sr=config.sample_rate, mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            S_db = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(6, 4))
            img = ax.imshow(S_db, origin="lower", aspect="auto")
            fig.colorbar(img, ax=ax)
            ax.set_title(f"Mel spectrogram - {cls_name}\n{slice_file_name}")
            ax.set_xlabel("Time frames")
            ax.set_ylabel("Mel bands")
            fig.tight_layout()

            out_name = f"example_spec_{cls_name}_{Path(slice_file_name).stem}.png"
            fig.savefig(results_root / out_name, dpi=200)
            plt.close(fig)

    print("Example spectrograms exported.")


# ------------------------------------------------
# Script Entry Point
# ------------------------------------------------
def main():
    config = ExperimentConfig(
        folds_to_use=list(range(1, 11)),  # use all 10 folds
        classes_to_use=None,              # use all classes in these folds
        max_files_per_class=120,
        test_size=0.2,
        val_size=0.2,
        random_seed=42,
        use_cross_val=True,
        cv_folds=5,
        evaluate_svm=True,
        evaluate_logreg=True,
        evaluate_rf=True,
    )
    t0 = time.perf_counter()
    run_experiment(config)
    elapsed = format_seconds(time.perf_counter() - t0)
    print(f"\nTotal runtime: {elapsed}")


if __name__ == "__main__":
    main()
