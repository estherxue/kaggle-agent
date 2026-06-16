"""Feature engineering for S6E6."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import CAT_COLS, ID_COL, TARGET_COL

FEATURE_SETS = ("base", "color", "color_redshift", "all_no_coord_bin", "all")


def load_data(data_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample


def _add_coord_features(out: pd.DataFrame, use_coords: bool) -> pd.DataFrame:
    if use_coords and {"alpha", "delta"}.issubset(out.columns):
        ra = out["alpha"].astype(float)
        dec = out["delta"].astype(float)
        out["alpha_delta"] = ra * dec
        out["sin_alpha"] = np.sin(np.radians(ra))
        out["cos_alpha"] = np.cos(np.radians(ra))
        out["sin_delta"] = np.sin(np.radians(dec))
        out["cos_delta"] = np.cos(np.radians(dec))
    return out


def _add_coord_bin(out: pd.DataFrame, use_coords: bool) -> pd.DataFrame:
    if use_coords and {"alpha", "delta"}.issubset(out.columns):
        ra = out["alpha"].astype(float)
        dec = out["delta"].astype(float)
        try:
            out["alpha_delta_bin"] = (
                pd.cut(ra, bins=20, labels=False) * 20 + pd.cut(dec, bins=20, labels=False)
            )
        except ValueError:
            pass
    return out


def _add_color_indices(out: pd.DataFrame) -> pd.DataFrame:
    if all(c in out.columns for c in ["u", "g"]):
        out["u_g"] = out["u"] - out["g"]
    if all(c in out.columns for c in ["g", "r"]):
        out["g_r"] = out["g"] - out["r"]
    if all(c in out.columns for c in ["r", "i"]):
        out["r_i"] = out["r"] - out["i"]
    if all(c in out.columns for c in ["i", "z"]):
        out["i_z"] = out["i"] - out["z"]
    return out


def _add_band_ratios(out: pd.DataFrame) -> pd.DataFrame:
    bands = [c for c in ["u", "g", "r", "i", "z"] if c in out.columns]
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            a, b = bands[i], bands[j]
            out[f"ratio_{a}_{b}"] = out[a] / (out[b].abs() + 1e-6)
    return out


def _add_redshift_basic(out: pd.DataFrame) -> pd.DataFrame:
    if "redshift" in out.columns:
        rz = out["redshift"].astype(float)
        out["redshift_bin"] = pd.qcut(rz, q=10, duplicates="drop", labels=False)
    return out


def _add_redshift_extended(out: pd.DataFrame) -> pd.DataFrame:
    if "redshift" not in out.columns:
        return out
    rz = out["redshift"].astype(float)
    out["log1p_abs_redshift"] = np.log1p(np.abs(rz))
    if "redshift_bin" not in out.columns:
        out["redshift_bin"] = pd.qcut(rz, q=10, duplicates="drop", labels=False)
    if "u_g" in out.columns:
        out["redshift_u_g"] = rz * out["u_g"]
    if "g_r" in out.columns:
        out["redshift_g_r"] = rz * out["g_r"]
    return out


def add_features(
    df: pd.DataFrame,
    feature_set: str = "all",
    use_coords: bool = True,
) -> pd.DataFrame:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set!r}; choose from {FEATURE_SETS}")

    out = df.copy()

    if feature_set == "base":
        out = _add_coord_features(out, use_coords=use_coords)
        out = _add_band_ratios(out)
        out = _add_redshift_basic(out)
        return out

    if feature_set in ("color", "color_redshift"):
        out = _add_coord_features(out, use_coords=use_coords)
        out = _add_band_ratios(out)
        out = _add_color_indices(out)
        out = _add_redshift_basic(out)
        if feature_set == "color_redshift":
            out = _add_redshift_extended(out)
        return out

    # all / all_no_coord_bin
    out = _add_color_indices(out)
    out = _add_coord_features(out, use_coords=use_coords)
    if feature_set == "all":
        out = _add_coord_bin(out, use_coords=use_coords)
    out = _add_redshift_extended(out)
    return out


def frequency_encode(
    train: pd.DataFrame, test: pd.DataFrame, col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = train[col].astype(str).value_counts(normalize=True)
    train = train.copy()
    test = test.copy()
    train[f"{col}_freq"] = train[col].astype(str).map(freq).fillna(0.0)
    test[f"{col}_freq"] = test[col].astype(str).map(freq).fillna(0.0)
    return train, test


def _label_encode_cats(train_f: pd.DataFrame, test_f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in CAT_COLS:
        if col in train_f.columns:
            le = LabelEncoder()
            combined = pd.concat([train_f[col], test_f[col]], axis=0).astype(str)
            le.fit(combined)
            train_f[col] = le.transform(train_f[col].astype(str))
            test_f[col] = le.transform(test_f[col].astype(str))
    return train_f, test_f


def prepare_lgb_xgb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str = "all",
    use_coords: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_f = add_features(train, feature_set=feature_set, use_coords=use_coords)
    test_f = add_features(test, feature_set=feature_set, use_coords=use_coords)

    use_freq = feature_set in ("all", "all_no_coord_bin")
    if use_freq:
        for col in CAT_COLS:
            if col in train_f.columns:
                train_f, test_f = frequency_encode(train_f, test_f, col)
    train_f, test_f = _label_encode_cats(train_f, test_f)

    if use_freq and "redshift_bin" in train_f.columns and "spectral_type_freq" in train_f.columns:
        train_f["spec_redshift_bin"] = (
            train_f["spectral_type_freq"] * train_f["redshift_bin"].fillna(-1)
        )
        test_f["spec_redshift_bin"] = (
            test_f["spectral_type_freq"] * test_f["redshift_bin"].fillna(-1)
        )

    drop_cols = {ID_COL, TARGET_COL}
    feature_cols = [
        c
        for c in train_f.columns
        if c not in drop_cols and c in test_f.columns and train_f[c].dtype != object
    ]
    return train_f, test_f, feature_cols


def prepare_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str = "all",
    use_coords: bool = True,
    cat_native: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_f = add_features(train, feature_set=feature_set, use_coords=use_coords)
    test_f = add_features(test, feature_set=feature_set, use_coords=use_coords)

    use_freq = feature_set in ("all", "all_no_coord_bin")
    if use_freq:
        for col in CAT_COLS:
            if col in train_f.columns:
                train_f, test_f = frequency_encode(train_f, test_f, col)

    if cat_native:
        for col in CAT_COLS:
            if col in train_f.columns:
                train_f[col] = train_f[col].astype(str)
                test_f[col] = test_f[col].astype(str)
        cat_feature_names = [c for c in CAT_COLS if c in train_f.columns]
    else:
        train_f, test_f = _label_encode_cats(train_f, test_f)
        cat_feature_names = []

    drop_cols = {ID_COL, TARGET_COL}
    num_cols = [
        c
        for c in train_f.columns
        if c not in drop_cols
        and c not in cat_feature_names
        and c in test_f.columns
        and train_f[c].dtype != object
    ]
    feature_cols = num_cols + cat_feature_names
    return train_f, test_f, feature_cols, cat_feature_names
