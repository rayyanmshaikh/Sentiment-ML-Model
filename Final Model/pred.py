import json
import re
from collections import Counter

import numpy as np
import pandas as pd

TEXT_COLS = ["feel_text", "food", "soundtrack"]
DATA_PATH = "cleaned_dataset.csv"
LOGREG_PARAMS_PATH = "logreg_params.npz"
LOGREG_TFIDF_PATH = "logreg_tfidf.json"
NB_PARAMS_PATH = "bernoulli_params.npz"
NB_VOCAB_PATH = "bernoulli_vocab.json"
STACK_META_PATH = "stack_meta_params.npz"
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

PAINTING_NAME_BY_LABEL = {
    0: "The Persistence of Memory",
    1: "The Starry Night",
    2: "The Water Lily Pond",
}


# Preprocessing

def rename_raw_columns(input_df):
    columns = [
        "id", "painting", "intensity", "feel_text", "sombre", "content", "calm", "uneasy",
        "n_colors", "n_objects", "pay", "room", "view_with", "season", "food", "soundtrack",
    ]
    return input_df.rename(columns=dict(zip(input_df.columns[: len(columns)], columns)))


def clean_string(value):
    if pd.isna(value):
        return None

    value = str(value).replace("-", " ").replace(",", " ").replace(".", " ")
    cleaned = re.sub(r"[^a-zA-Z ]", "", value).strip().lower()
    return cleaned if cleaned else None


def extract_numeric(value):
    if pd.isna(value):
        return None
    match = re.match(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def extract_money(value):
    if pd.isna(value):
        return None

    text = str(value).lower().strip()
    if "million" in text:
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match:
            return int(float(match.group(1)) * 1_000_000)
        return None

    cleaned = re.sub(r"[^\d]", "", text)
    return int(cleaned) if cleaned.isdigit() else None


PAY_LOG_CLIP = 19.599335743370936


def transform_pay_for_inference(pay_series):
    pay = pd.to_numeric(pay_series, errors="coerce")
    pay = pay.clip(lower=0)
    pay = np.log1p(pay)
    if PAY_LOG_CLIP is not None and np.isfinite(PAY_LOG_CLIP):
        pay = pay.clip(upper=PAY_LOG_CLIP)
    return pay


def multi_hot_encode(df, column, prefix):
    split_series = df[column].fillna("").apply(
        lambda x: [i.strip() for i in str(x).split(",") if i.strip()]
    )
    unique_values = sorted({item for sublist in split_series for item in sublist})
    for val in unique_values:
        col_name = f"{prefix}_{val.replace(' ', '_')}"
        df[col_name] = split_series.apply(lambda lst: int(val in lst))
    return df


def extract_text_features(text_series, prefix):
    features = pd.DataFrame(index=text_series.index)
    features[f"{prefix}_word_count"] = text_series.fillna("").apply(lambda x: len(str(x).split()))
    features[f"{prefix}_avg_word_len"] = text_series.fillna("").apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    return features


def preprocess_dataset(dataset):
    df = dataset.copy()

    # If dataset already looks cleaned, keep it unchanged.
    cleaned_markers = {
        "pay_missing", "n_colors_missing", "n_objects_missing",
        "feel_text_word_count", "food_word_count", "soundtrack_word_count",
    }
    if cleaned_markers.issubset(set(df.columns)):
        return df

    # Normalize raw names first.
    if not all(col in df.columns for col in ["feel_text", "food", "soundtrack"]):
        df = rename_raw_columns(df)

    missing_indicator_cols = [
        "pay", "n_colors", "n_objects", "feel_text", "food", "soundtrack", "room", "view_with", "season",
    ]
    for col in missing_indicator_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    if "pay" in df.columns:
        df["pay"] = df["pay"].apply(extract_money)
        df["pay"] = transform_pay_for_inference(df["pay"])

    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        text_features = extract_text_features(df[col], col)
        df = pd.concat([df, text_features], axis=1)
        df[col] = df[col].apply(clean_string)

    for col in ["sombre", "content", "calm", "uneasy"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)

    for col in ["n_colors", "n_objects", "intensity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["room", "view_with", "season"]:
        if col in df.columns:
            df = multi_hot_encode(df, col, prefix=col)
            df = df.drop(columns=[col])

    if "painting" in df.columns:
        painting_mapping = {
            name: idx for idx, name in enumerate(sorted(df["painting"].dropna().unique()))
        }
        df["target"] = df["painting"].map(painting_mapping)

    return df


# Model loading and inference helpers

def fill_missing_text(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")


def required_columns(df, numeric_cols, text_cols):
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
    return df


def load_logreg(params_path=LOGREG_PARAMS_PATH, tfidf_path=LOGREG_TFIDF_PATH):
    params = np.load(params_path, allow_pickle=True)
    with open(tfidf_path, "r", encoding="utf-8") as f:
        tfidf_meta = json.load(f)

    num_stds = np.asarray(params["num_stds"], dtype=np.float64)
    num_stds = np.where(np.abs(num_stds) < 1e-12, 1.0, num_stds)

    text_specs = [
        (
            col,
            {word: int(idx) for word, idx in tfidf_meta[col]["vocab"].items()},
            np.asarray(tfidf_meta[col]["idf"], dtype=np.float64),
        )
        for col in TEXT_COLS
    ]

    return {
        "coef": np.asarray(params["coef"], dtype=np.float64),
        "intercept": np.asarray(params["intercept"], dtype=np.float64).ravel(),
        "class_order": np.asarray(params["class_order"]),
        "numeric_cols": params["numeric_cols"].tolist(),
        "num_means": np.asarray(params["num_means"], dtype=np.float64),
        "num_stds": num_stds,
        "test_indices": np.asarray(params["test_indices"], dtype=np.int64),
        "text_specs": text_specs,
    }


def load_nb(params_path=NB_PARAMS_PATH, vocab_path=NB_VOCAB_PATH):
    params = np.load(params_path, allow_pickle=True)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    return {
        "class_order": np.asarray(params["class_order"]),
        "class_log_prior": np.asarray(params["class_log_prior"], dtype=np.float64),
        "feature_log_prob": np.asarray(params["feature_log_prob"], dtype=np.float64),
        "neg_log_prob": np.asarray(params["neg_log_prob"], dtype=np.float64),
        "vocab_data": vocab_data,
    }


def load_stack(path=STACK_META_PATH):
    params = np.load(path, allow_pickle=True)
    return {
        "coef": np.asarray(params["coef"], dtype=np.float64),
        "intercept": np.asarray(params["intercept"], dtype=np.float64).ravel(),
        "class_order": np.asarray(params["class_order"]),
        "bernoulli_class_order": np.asarray(params["bernoulli_class_order"]),
        "logreg_class_order": np.asarray(params["logreg_class_order"]),
    }


# Feature building

def build_binary_matrix(texts, vocab):
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in str(text).split():
            if word in vocab:
                X[i, vocab[word]] = 1.0
    return X


def build_nb_features(df, vocab_data):
    blocks = []
    for col in TEXT_COLS:
        vocab = {word: int(idx) for word, idx in vocab_data[col]["vocab"].items()}
        blocks.append(build_binary_matrix(df[col], vocab))
    return np.hstack(blocks)


def build_tfidf(text_series, vocab, idf_vec):
    block = np.zeros((len(text_series), len(vocab)), dtype=np.float64)
    for i, text in enumerate(text_series):
        tokens = TOKEN_RE.findall(str(text).lower())
        if not tokens:
            continue

        counts = Counter(tok for tok in tokens if tok in vocab)
        if not counts:
            continue

        total = float(sum(counts.values()))
        norm_sq = 0.0
        for tok, count in counts.items():
            j = vocab[tok]
            val = (count / total) * idf_vec[j]
            block[i, j] = val
            norm_sq += val * val

        norm = np.sqrt(norm_sq)
        if norm > 0.0:
            block[i, :] /= norm
    return block


# LOGISTIC REGRESSION MODEL

def build_logreg_features(df, numeric_cols, num_means, num_stds, text_specs):
    X_num = df[numeric_cols].to_numpy(dtype=np.float64)
    X_num = np.where(np.isnan(X_num), num_means, X_num)
    X_num = (X_num - num_means) / num_stds
    text_blocks = [build_tfidf(df[col], vocab, idf_vec) for col, vocab, idf_vec in text_specs]
    return np.hstack([X_num] + text_blocks)


def predict_logreg_proba(X_mat, coef, intercept):
    logits = np.asarray(X_mat @ coef.T) + intercept.reshape(1, -1)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


# BERNOULLI NAIVE BAYES MODEL

def predict_bernoulli_nb_log_scores(X_bin, model):
    log_scores = np.tile(model["class_log_prior"], (X_bin.shape[0], 1))
    log_scores += X_bin @ model["feature_log_prob"].T
    log_scores += (1.0 - X_bin) @ model["neg_log_prob"].T
    return log_scores


def predict_bernoulli_nb_proba(X_bin, model):
    log_scores = predict_bernoulli_nb_log_scores(X_bin, model)
    log_scores = log_scores - np.max(log_scores, axis=1, keepdims=True)
    exp_scores = np.exp(log_scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def align_proba(proba, src, trgt):
    src = np.asarray(src).tolist()
    trgt = np.asarray(trgt).tolist()
    index_lookup = {label: idx for idx, label in enumerate(src)}
    return np.column_stack([proba[:, index_lookup[label]] for label in trgt])


# STACKED META-MODEL PREDICTION
def predict_labels_from_clean(clean):
    df = clean.copy()
    df = required_columns(df, logreg_model["numeric_cols"], TEXT_COLS)
    fill_missing_text(df, TEXT_COLS)

    nb_features = build_nb_features(df, nb_model["vocab_data"])
    logreg_features = build_logreg_features(
        df,
        logreg_model["numeric_cols"],
        logreg_model["num_means"],
        logreg_model["num_stds"],
        logreg_model["text_specs"],
    )

    nb_proba = align_proba(
        predict_bernoulli_nb_proba(nb_features, nb_model),
        nb_model["class_order"],
        stack_model["bernoulli_class_order"],
    )
    logreg_proba = align_proba(
        predict_logreg_proba(logreg_features, logreg_model["coef"], logreg_model["intercept"]),
        logreg_model["class_order"],
        stack_model["logreg_class_order"],
    )

    stack_pred = stack_model["class_order"][
        np.argmax(
            predict_logreg_proba(
                np.hstack([nb_proba, logreg_proba]),
                stack_model["coef"],
                stack_model["intercept"],
            ),
            axis=1,
        )
    ]
    return np.asarray(stack_pred)


def labels_to_painting_names(labels):
    return [PAINTING_NAME_BY_LABEL.get(int(label), str(label)) for label in labels]


# LOAD TRAINED MODEL PARAMETERS
logreg_model = load_logreg()
nb_model = load_nb()
stack_model = load_stack()


def predict_all(dataset):
    return labels_to_painting_names(predict_labels_from_clean(preprocess_dataset(dataset)))
