import os
import csv
import re
import argparse
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import numpy as np
import sys

# a_numpy.py
# GitHub Copilot
# MovieLens ml-latest-small analysis script using numpy (no pandas)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_csv_safe(path: str) -> Optional[List[Dict[str, str]]]:
    if not os.path.exists(path):
        logging.warning("File not found: %s", path)
        return None
    try:
        with open(path, newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            return [dict(row) for row in rdr]
    except Exception as e:
        logging.error("Failed to read %s: %s", path, e)
        return None


def parse_year_from_title(title: Optional[str]) -> Optional[int]:
    if not isinstance(title, str):
        return None
    m = re.search(r"\((\d{4})\)\s*$", title)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def clean_movies(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, Any]]:
    required = {"movieId", "title", "genres"}
    if not rows:
        raise ValueError("movies.csv is empty")
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"movies.csv missing required columns: {required - set(rows[0].keys())}")
    movies = {}
    for r in rows:
        try:
            mid = int(r["movieId"])
        except Exception:
            continue
        title = r.get("title") or "Unknown Title"
        genres_raw = r.get("genres") or ""
        if genres_raw == "(no genres listed)":
            genres_raw = ""
        genre_list = [] if genres_raw == "" else genres_raw.split("|")
        year = parse_year_from_title(title)
        movies[mid] = {"movieId": mid, "title": title, "genres": genres_raw, "genre_list": genre_list, "year": year}
    return movies


def clean_ratings(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    required = {"userId", "movieId", "rating", "timestamp"}
    if not rows:
        raise ValueError("ratings.csv is empty")
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"ratings.csv missing required columns: {required - set(rows[0].keys())}")
    seen = set()
    clean = []
    for r in rows:
        key = (r.get("userId"), r.get("movieId"), r.get("timestamp"))
        if key in seen:
            continue
        seen.add(key)
        try:
            userId = int(r["userId"])
            movieId = int(r["movieId"])
        except Exception:
            continue
        try:
            rating = float(r["rating"])
        except Exception:
            continue
        if not (0.5 <= rating <= 5.0):
            continue
        try:
            ts = float(r["timestamp"])
            if ts < 0:
                continue
        except Exception:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        clean.append({"userId": userId, "movieId": movieId, "rating": rating, "timestamp": ts, "datetime": dt})
    return clean


def merge_data(movies: Dict[int, Dict[str, Any]], ratings: List[Dict[str, Any]], tags: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    tag_map = {}
    if tags:
        # aggregate unique tags per movieId
        for t in tags:
            try:
                mid = int(t.get("movieId"))
            except Exception:
                continue
            tag = t.get("tag")
            if tag is None or tag == "":
                continue
            tag_set = tag_map.setdefault(mid, set())
            tag_set.add(tag)
        # convert to semicolon separated
        tag_map = {k: ";".join(sorted(v)) for k, v in tag_map.items()}

    merged = []
    for r in ratings:
        mid = r["movieId"]
        m = movies.get(mid, {"movieId": mid, "title": "", "genres": "", "genre_list": [], "year": None})
        row = {
            "userId": r["userId"],
            "movieId": mid,
            "rating": r["rating"],
            "timestamp": r["timestamp"],
            "datetime": r["datetime"],
            "title": m["title"],
            "genres": m["genres"],
            "genre_list": m["genre_list"],
            "year": m["year"],
            "tags": tag_map.get(mid, "")
        }
        merged.append(row)
    return merged


def filter_by_genre(merged: List[Dict[str, Any]], genre: Optional[str]) -> List[Dict[str, Any]]:
    if not genre:
        return merged
    return [r for r in merged if genre in r.get("genre_list", [])]


def filter_by_year_range(merged: List[Dict[str, Any]], start_year: Optional[int], end_year: Optional[int]) -> List[Dict[str, Any]]:
    if start_year is None and end_year is None:
        return merged
    out = []
    for r in merged:
        y = r.get("year")
        if y is None:
            continue
        if start_year is not None and y < start_year:
            continue
        if end_year is not None and y > end_year:
            continue
        out.append(r)
    return out


def compute_movie_statistics(merged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[int, List[float]] = {}
    meta: Dict[int, Dict[str, Any]] = {}
    for r in merged:
        mid = r["movieId"]
        groups.setdefault(mid, []).append(r["rating"])
        if mid not in meta:
            meta[mid] = {"title": r.get("title", ""), "year": r.get("year", None)}
    stats = []
    for mid, ratings in groups.items():
        arr = np.array(ratings, dtype=float)
        count = arr.size
        mean = float(np.mean(arr)) if count > 0 else float("nan")
        median = float(np.median(arr)) if count > 0 else float("nan")
        std = float(np.std(arr, ddof=1)) if count > 1 else float("nan")
        stats.append({
            "movieId": mid,
            "title": meta[mid]["title"],
            "year": meta[mid]["year"],
            "rating_count": int(count),
            "rating_mean": mean,
            "rating_median": median,
            "rating_std": std
        })
    # sort by rating_count desc, then rating_mean desc
    stats.sort(key=lambda x: ( -x["rating_count"], - (x["rating_mean"] if not np.isnan(x["rating_mean"]) else -np.inf)))
    return stats


def compute_genre_statistics(merged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ggroups: Dict[str, List[float]] = {}
    for r in merged:
        for g in r.get("genre_list", []):
            if not g:
                continue
            ggroups.setdefault(g, []).append(r["rating"])
    out = []
    for g, ratings in ggroups.items():
        arr = np.array(ratings, dtype=float)
        count = arr.size
        mean = float(np.mean(arr)) if count > 0 else float("nan")
        median = float(np.median(arr)) if count > 0 else float("nan")
        out.append({"genre": g, "rating_count": int(count), "rating_mean": mean, "rating_median": median})
    out.sort(key=lambda x: -x["rating_count"])
    return out


def ratings_time_stats(ratings: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ratings:
        return {"min_datetime": None, "max_datetime": None, "total_ratings": 0}
    dts = [r["datetime"] for r in ratings]
    return {"min_datetime": min(dts), "max_datetime": max(dts), "total_ratings": len(dts)}


def missing_summary_rows(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    if not rows:
        return {}
    keys = rows[0].keys()
    missing = {k: 0 for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                missing[k] += 1
    return missing


def save_csv(out_path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not rows:
        # write header if provided
        if fieldnames:
            with open(out_path, "w", newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return
    if fieldnames is None:
        # stable order
        fieldnames = list(rows[0].keys())
    try:
        with open(out_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                # convert non-serializable types
                row = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
                writer.writerow(row)
        logging.info("Saved %s", out_path)
    except Exception as e:
        logging.error("Failed to save %s: %s", out_path, e)


def main(data_dir: str, out_dir: str, min_count: int, genre: Optional[str], start_year: Optional[int], end_year: Optional[int]):
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")
    tags_path = os.path.join(data_dir, "tags.csv")

    movies_raw = read_csv_safe(movies_path)
    ratings_raw = read_csv_safe(ratings_path)
    tags_raw = read_csv_safe(tags_path)

    # If required files are missing, fall back to a minimal in-memory sample dataset
    if movies_raw is None or ratings_raw is None:
        logging.warning("Required CSV files not found in %s. Falling back to minimal in-memory sample dataset.", data_dir)
        if movies_raw is None:
            movies_raw = [
                {"movieId": "1", "title": "Sample Movie (2020)", "genres": "Drama"}
            ]
        if ratings_raw is None:
            # timestamp 1609459200 == 2021-01-01T00:00:00Z
            ratings_raw = [
                {"userId": "1", "movieId": "1", "rating": "4.0", "timestamp": "1609459200"}
            ]
        if tags_raw is None:
            tags_raw = []

    movies = clean_movies(movies_raw)
    ratings = clean_ratings(ratings_raw)
    merged = merge_data(movies, ratings, tags_raw)

    logging.info("Movies: %d unique, Ratings: %d rows, Merged: %d rows", len(movies), len(ratings), len(merged))

    if genre:
        merged = filter_by_genre(merged, genre)
        logging.info("Filtered merged dataset by genre '%s': %d rows", genre, len(merged))
    merged = filter_by_year_range(merged, start_year, end_year)
    if start_year or end_year:
        logging.info("Filtered merged dataset by year range %s - %s: %d rows", start_year, end_year, len(merged))

    movie_stats = compute_movie_statistics(merged)
    genre_stats = compute_genre_statistics(merged)
    time_stats = ratings_time_stats(ratings)
    missing_movies = missing_summary_rows(list(movies.values()))
    missing_ratings = missing_summary_rows(ratings)

    top_movies = [m for m in movie_stats if m["rating_count"] >= min_count]
    top_movies.sort(key=lambda x: (-x["rating_mean"], -x["rating_count"]))
    top_movies = top_movies[:50]

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    save_csv(os.path.join(out_dir, "merged.csv"), merged,
             fieldnames=["userId", "movieId", "rating", "timestamp", "datetime", "title", "genres", "tags", "year"])
    save_csv(os.path.join(out_dir, "movie_stats.csv"), movie_stats)
    save_csv(os.path.join(out_dir, "genre_stats.csv"), genre_stats)
    save_csv(os.path.join(out_dir, "top_movies.csv"), top_movies)

    # Print short summary
    print("\nSummary")
    print("-------")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Total movies: {len(movies)}")
    print(f"Total ratings: {len(ratings)}")
    print(f"Merged rows after filtering: {len(merged)}")
    print("\nRatings time range:")
    print(time_stats)
    print("\nTop genres (by rating count):")
    for g in genre_stats[:10]:
        print(f"{g['genre']}: count={g['rating_count']}, mean={g['rating_mean']:.3f}")
    print("\nTop movies (min rating count >= {0}):".format(min_count))
    for m in top_movies[:20]:
        print(f"{m['movieId']} | {m['title']} ({m['year']}) count={m['rating_count']} mean={m['rating_mean']:.3f}")
    print("\nMissing values in movies.csv:")
    print(missing_movies)
    print("\nMissing values in ratings.csv:")
    print(missing_ratings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens ml-latest-small analysis (numpy version)")
    parser.add_argument("--data-dir", default=None, help="Path to extracted ml-latest-small directory containing movies.csv, ratings.csv (and optional tags.csv). If not provided, common locations and ML_DATA_DIR env var will be checked.")
    parser.add_argument("--out-dir", default="ml_outputs", help="Directory to save analysis CSV outputs")
    parser.add_argument("--min-count", type=int, default=20, help="Minimum number of ratings to consider a movie for top list")
    parser.add_argument("--genre", default="", help="Optional: filter by single genre (e.g., Comedy)")
    parser.add_argument("--start-year", type=int, default=None, help="Optional: filter movies from this year (inclusive)")
    parser.add_argument("--end-year", type=int, default=None, help="Optional: filter movies up to this year (inclusive)")

    # Use parse_known_args to avoid ipykernel / Jupyter CLI noise; then ensure data_dir is set or auto-detected
    args, _ = parser.parse_known_args()

    # Normalize/validate provided data_dir; try auto-detect if missing or invalid.
    if args.data_dir is None or not os.path.isdir(args.data_dir):
        # try environment variable and a few common relative paths
        candidates = [
            os.environ.get("ML_DATA_DIR"),
            "./ml-latest-small",
            "./data/ml-latest-small",
            "../ml-latest-small"
        ]
        data_dir = next((p for p in candidates if p and os.path.isdir(p)), None)
        if data_dir:
            args.data_dir = data_dir
            logging.info("Auto-detected data directory: %s", data_dir)
        else:
            # In interactive sessions (e.g. Jupyter / IPython) avoid exiting the process;
            # fall back to current working directory and let main() report missing files.
            interactive = ("ipykernel" in sys.modules) or hasattr(sys, "ps1")
            if interactive:
                fallback = os.getcwd()
                args.data_dir = fallback
                logging.warning("No data directory found; running in interactive mode - falling back to current working directory: %s", fallback)
            else:
                parser.print_help()
                logging.error("No --data-dir provided and no data directory found in default locations. Set ML_DATA_DIR or provide --data-dir.")
                sys.exit(1)

    main(args.data_dir, args.out_dir, args.min_count, args.genre or None, args.start_year, args.end_year)