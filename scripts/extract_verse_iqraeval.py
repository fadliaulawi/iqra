import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Text normalization utilities
# -----------------------------

ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

TATWEEL_RE = re.compile(r"\u0640")
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s\u0600-\u06FF]")

# Optional letter normalization map
CHAR_MAP = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
    "ة": "ه",   # can be changed if needed
})


def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text)


def normalize_arabic(
    text: Any,
    *,
    remove_diacritics: bool = False,
    normalize_chars: bool = True,
    remove_punct: bool = True,
) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = unicodedata.normalize("NFKC", text)
    text = TATWEEL_RE.sub("", text)

    if remove_diacritics:
        text = strip_diacritics(text)

    if normalize_chars:
        text = text.translate(CHAR_MAP)

    if remove_punct:
        text = PUNCT_RE.sub(" ", text)

    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# -----------------------------
# Qur'an corpus loading
# -----------------------------

def load_quran_corpus(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    required = {"surah_id", "ayah_id", "text"}
    for i, row in enumerate(corpus):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Corpus row {i} missing keys: {missing}")

    return corpus


def build_indexes(corpus: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    exact_index: Dict[str, List[Dict[str, Any]]] = {}
    nodiac_index: Dict[str, List[Dict[str, Any]]] = {}

    for row in corpus:
        raw_text = row["text"]
        exact_key = normalize_arabic(
            raw_text,
            remove_diacritics=False,
            normalize_chars=True,
            remove_punct=True,
        )
        nodiac_key = normalize_arabic(
            raw_text,
            remove_diacritics=True,
            normalize_chars=True,
            remove_punct=True,
        )

        exact_index.setdefault(exact_key, []).append(row)
        nodiac_index.setdefault(nodiac_key, []).append(row)

    return {
        "exact": exact_index,
        "nodiac": nodiac_index,
    }


# -----------------------------
# Matching logic
# -----------------------------

def find_best_match(
    verse_text: str,
    indexes: Dict[str, Dict[str, List[Dict[str, Any]]]],
    corpus: List[Dict[str, Any]],
    fuzzy_threshold: float = 0.85,
) -> Dict[str, Any]:
    result = {
        "surah_id": None,
        "ayah_id": None,
        "quran_text_matched": None,
        "match_type": "no_match",
        "match_score": 0.0,
    }

    if not verse_text or not str(verse_text).strip():
        result["match_type"] = "empty_input"
        return result

    query_exact = normalize_arabic(
        verse_text,
        remove_diacritics=False,
        normalize_chars=True,
        remove_punct=True,
    )
    query_nodiac = normalize_arabic(
        verse_text,
        remove_diacritics=True,
        normalize_chars=True,
        remove_punct=True,
    )

    # 1) Exact normalized match with diacritics retained
    exact_hits = indexes["exact"].get(query_exact)
    if exact_hits:
        hit = exact_hits[0]
        result.update({
            "surah_id": hit["surah_id"],
            "ayah_id": hit["ayah_id"],
            "quran_text_matched": hit["text"],
            "match_type": "exact",
            "match_score": 1.0,
        })
        return result

    # 2) Exact normalized match after diacritic stripping
    nodiac_hits = indexes["nodiac"].get(query_nodiac)
    if nodiac_hits:
        hit = nodiac_hits[0]
        result.update({
            "surah_id": hit["surah_id"],
            "ayah_id": hit["ayah_id"],
            "quran_text_matched": hit["text"],
            "match_type": "normalized_no_diacritics",
            "match_score": 0.99,
        })
        return result

    # 3) Fuzzy fallback on no-diacritics normalized text
    best_row: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for row in corpus:
        corpus_text = normalize_arabic(
            row["text"],
            remove_diacritics=True,
            normalize_chars=True,
            remove_punct=True,
        )
        score = similarity(query_nodiac, corpus_text)
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is not None and best_score >= fuzzy_threshold:
        result.update({
            "surah_id": best_row["surah_id"],
            "ayah_id": best_row["ayah_id"],
            "quran_text_matched": best_row["text"],
            "match_type": "fuzzy",
            "match_score": round(best_score, 4),
        })
        return result

    return result


# -----------------------------
# Main dataframe enrichment
# -----------------------------

def enrich_iqraeval_with_verse_metadata(
    df: pd.DataFrame,
    quran_json_path: str,
    text_col_priority: Tuple[str, ...] = ("tashkeel", "sentence"),
    fuzzy_threshold: float = 0.85,
) -> pd.DataFrame:
    corpus = load_quran_corpus(quran_json_path)
    indexes = build_indexes(corpus)

    df = df.copy()

    # choose source text column
    source_col = None
    for col in text_col_priority:
        if col in df.columns:
            source_col = col
            break

    if source_col is None:
        raise ValueError(
            f"None of the text columns were found: {text_col_priority}. "
            f"Available columns: {list(df.columns)}"
        )

    records = []
    for text in df[source_col].tolist():
        match = find_best_match(
            verse_text=text,
            indexes=indexes,
            corpus=corpus,
            fuzzy_threshold=fuzzy_threshold,
        )
        records.append(match)

    meta_df = pd.DataFrame(records)

    # add columns back
    for col in meta_df.columns:
        df[col] = meta_df[col]

    df["verse_key"] = df.apply(
        lambda x: (
            f"{int(x['surah_id']):03d}:{int(x['ayah_id']):03d}"
            if pd.notna(x["surah_id"]) and pd.notna(x["ayah_id"])
            else None
        ),
        axis=1,
    )

    return df


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    import requests

    def get_quran():
        verses = []
        page = 1

        while True:
            url = f"https://api.quran.com/api/v4/verses/uthmani?page={page}&per_page=50"
            res = requests.get(url).json()

            for v in res["verses"]:
                verses.append({
                    "surah_id": v["chapter_id"],
                    "ayah_id": v["verse_number"],
                    "text": v["text_uthmani"]
                })

            if not res["pagination"]["next_page"]:
                break

            page += 1
            print(f"Downloaded page {page}")

        return verses

    quran = get_quran()

    # Example: load your dataset CSV/Parquet
    # df = pd.read_csv("data/raw/train/metadata.csv")

    # enriched_df = enrich_iqraeval_with_verse_metadata(
    #     df=df,
    #     quran_json_path="quran_corpus.json",
    #     text_col_priority=("text",),
    #     fuzzy_threshold=0.86,
    # )

    # print(enriched_df.head())

    # Save back
    # enriched_df.to_parquet("iqra_train_with_verse_meta.parquet", index=False)
    # enriched_df.to_csv("iqra_train_with_verse_meta.csv", index=False)