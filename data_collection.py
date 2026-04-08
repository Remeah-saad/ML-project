"""
data_collection.py
==================
Collects Arabic-language Google Play reviews from the Saudi Arabian store
for a curated list of apps spanning multiple sectors.

Design decisions (per project spec):
- country='sa'  → Saudi Arabian Play Store only
- lang='ar'     → Arabic reviews only
- Per-star iteration (1–5) → prevents majority-class bias by fetching
  up to MAX_PER_STAR reviews for EACH rating level independently
- UTF-8-SIG encoding → preserves Arabic script in CSV output
- Apps span: Government, Finance, Food Delivery, Social Media, AI

Usage:
    python src/data_collection.py                  # defaults
    python src/data_collection.py --per-star 200 --out data/reviews.csv
"""

import time
import logging
import argparse
import pandas as pd
from pathlib import Path
from google_play_scraper import app as get_app_meta, reviews, Sort
from google_play_scraper.exceptions import NotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Target applications (per project specification) ───────────────────────────
TARGET_APPS = [
    # (package_id,            display_name,        sector)
    ("com.absher.services",   "Absher",             "Government"),
    ("com.stcpay",            "STC Pay",            "Finance"),
    ("com.hungerstation.app", "HungerStation",      "Food Delivery"),
    ("com.whatsapp",          "WhatsApp",           "Social Media"),
    ("com.stc.solutions",     "MySTC",              "Telecom"),
    ("com.snapchat.android",  "Snapchat",           "Social Media"),
]

COUNTRY     = "sa"          # Saudi Arabian Play Store
LANGUAGE    = "ar"          # Arabic reviews only
SORT_ORDER  = Sort.NEWEST


# ── Stage 1: App metadata ──────────────────────────────────────────────────────
def fetch_app_metadata(app_id: str) -> dict:
    """
    Fetches app-level details: title, category, overall rating, install count.
    These provide context for each review (as described in spec Stage 1).
    """
    try:
        meta = get_app_meta(app_id, lang=LANGUAGE, country=COUNTRY)
        return {
            "appId":     app_id,
            "title":     meta.get("title", ""),
            "category":  meta.get("genre", ""),
            "rating":    meta.get("score", 0.0),
            "installs":  meta.get("installs", ""),
        }
    except NotFoundError:
        log.warning(f"App not found: {app_id}")
        return {}
    except Exception as e:
        log.warning(f"Metadata error for {app_id}: {e}")
        return {}


# ── Stage 2: Per-star review iteration ────────────────────────────────────────
def fetch_reviews_for_star(
    app_id: str,
    star: int,
    max_count: int = 200,
    sleep_sec: float = 1.0,
) -> list[dict]:
    """
    Fetches up to `max_count` reviews for a single star rating.
    Iterating per star (1–5) is the key technique to avoid majority-class bias,
    ensuring the dataset has sufficient negative examples for the model to learn.
    """
    collected  = []
    token      = None
    batch_size = min(200, max_count)

    while len(collected) < max_count:
        remaining = max_count - len(collected)
        try:
            result, token = reviews(
                app_id,
                lang=LANGUAGE,
                country=COUNTRY,
                sort=SORT_ORDER,
                count=min(batch_size, remaining),
                filter_score_with=star,        # ← key: filter by star rating
                continuation_token=token,
            )
        except Exception as e:
            log.warning(f"  Error fetching {app_id} star={star}: {e}")
            break

        if not result:
            break

        collected.extend(result)

        if token is None:
            break
        time.sleep(sleep_sec)

    return collected


def scrape_all_apps(max_per_star: int = 200) -> pd.DataFrame:
    """
    Main collection pipeline (Stages 1–3 from spec):
      Stage 1 → Metadata extraction per app
      Stage 2 → Nested loop: for each app × for each star (1–5)
      Stage 3 → Flatten JSON → DataFrame → export CSV

    Returns a combined DataFrame for all apps.
    """
    all_records = []

    for app_id, display_name, sector in TARGET_APPS:
        log.info(f"\n{'='*55}")
        log.info(f"App: {display_name}  ({sector})  [{app_id}]")
        log.info(f"{'='*55}")

        # Stage 1: app metadata
        meta = fetch_app_metadata(app_id)
        if not meta:
            log.warning(f"Skipping {display_name} – metadata unavailable.")
            continue

        log.info(
            f"  Title    : {meta['title']}\n"
            f"  Category : {meta['category']}\n"
            f"  Rating   : {meta['rating']:.2f}\n"
            f"  Installs : {meta['installs']}"
        )

        # Stage 2: per-star nested loop
        app_total = 0
        for star in range(1, 6):
            raw = fetch_reviews_for_star(app_id, star, max_per_star)
            log.info(f"  ★{star} → fetched {len(raw):>4} reviews")

            for r in raw:
                content = (r.get("content") or "").strip()
                if not content:          # skip empty reviews
                    continue

                all_records.append({
                    # Core review fields (Stage 3 spec)
                    "reviewId":       r.get("reviewId", ""),
                    "userName":       r.get("userName", ""),
                    "content":        content,
                    "score":          r.get("score", star),
                    "thumbsUpCount":  r.get("thumbsUpCount", 0),
                    "at":             r.get("at", ""),
                    "appVersion":     r.get("appVersion", ""),
                    "replyContent":   r.get("replyContent", ""),
                    # App-level context
                    "appId":          app_id,
                    "appName":        display_name,
                    "sector":         sector,
                    "appCategory":    meta["category"],
                    "appRating":      meta["rating"],
                    "appInstalls":    meta["installs"],
                })
                app_total += 1

        log.info(f"  → {app_total:,} valid reviews collected for {display_name}")

    # Stage 3: build DataFrame
    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset="reviewId", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"\nTotal dataset: {len(df):,} reviews across {df['appId'].nunique()} apps")
    _print_distribution(df)
    return df


def _print_distribution(df: pd.DataFrame):
    log.info("\nStar rating distribution:")
    for star, count in df["score"].value_counts().sort_index().items():
        bar = "█" * (count // 20)
        log.info(f"  ★{star}  {count:>5}  {bar}")
    log.info("\nApp distribution:")
    for app, count in df["appName"].value_counts().items():
        log.info(f"  {app:<20} {count:>5} reviews")


# ── Stage 3: Save to CSV ───────────────────────────────────────────────────────
def save_dataset(df: pd.DataFrame, out_path: str = "data/reviews.csv"):
    """
    Saves with UTF-8-SIG encoding to preserve Arabic script integrity
    (as specified in the project documentation).
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info(f"\nSaved {len(df):,} reviews → {path}  (UTF-8-SIG encoding)")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Scrape Arabic Google Play reviews from Saudi store"
    )
    parser.add_argument(
        "--per-star", type=int, default=200,
        help="Max reviews per star rating per app (default: 200, max: ~200)"
    )
    parser.add_argument(
        "--out", default="data/reviews.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    log.info(f"Collection config:")
    log.info(f"  Country  : {COUNTRY} (Saudi Arabia)")
    log.info(f"  Language : {LANGUAGE} (Arabic)")
    log.info(f"  Per star : {args.per_star} reviews")
    log.info(f"  Apps     : {len(TARGET_APPS)}")

    df = scrape_all_apps(max_per_star=args.per_star)
    save_dataset(df, args.out)


if __name__ == "__main__":
    main()
