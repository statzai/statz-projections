"""
Phase 4: weekly model retraining — dry-run mode.

Iterates every (league, stat) pair, trains a fresh PoissonRegressor on
the DB-sourced training data (projection_model_dataset, populated by
Phase 2 dual-write + Phase 3 reads), writes the .sav to a versioned path,
and INSERTs a row in projection_models with is_active=0.

For each (league, stat) ALSO logs what the promotion decision WOULD be
against the current incumbent, using the guardrail in
_decide_promotion(). Phase 4 does NOT actually promote — that's Phase 5.
Purpose of dry-run: observe week-over-week MAE variance for 3–4 cycles
so the 2%/10% thresholds can be tuned empirically before promotion goes
live.

Also trains the "All Leagues" fallback models (competition_id=NULL rows)
by training on every row in projection_model_dataset, regardless of
league.

Called from /api/retrain endpoint (routes.py). Not yet scheduled — Phase
5 adds the weekly cron on Laravel's Kernel.
"""
import gc
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from app.repository.projection_dataset_repo import load_model_dataset_async
from app.repository.projection_model_repo import fetch_active_model, insert_projection_model
from app.services.statz_functions import get_stat_list

logger = logging.getLogger("retrain")

# Minimum training rows to attempt a per-league model. Below this, we skip
# the league and let it fall back to the All Leagues pool at projection
# time. Current Premier League has ~1,600 rows, Liga Portugal ~1,400,
# smaller comps fewer — 100 is a defensive floor, NOT a quality threshold.
MIN_TRAINING_ROWS = 100

# Per-design: fit_model (direct PoissonRegressor newton-cholesky fit) for
# the two Passes stats because grid_search was historically slow + offered
# little improvement on the wider Passes distributions. grid_search for
# everything else.
_FIT_MODEL_STATS = ("Passes", "Successful Passes")


def _snake(stat_name: str) -> str:
    """'Shots On Target' → 'shots_on_target'."""
    return stat_name.lower().replace(" ", "_")


def _predictor_columns(stat_name: str) -> tuple:
    """Parquet-format column names returned by load_model_dataset_async."""
    return (f"Team {stat_name} History", f"Opponent {stat_name} History Against")


def _target_column(stat_name: str) -> str:
    return f"Team {stat_name}"


def _model_build_path(league_dir_name: str, stat_name: str) -> str:
    """Timestamped file path under /app/app/model-builds/{league_dir}/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep the existing directory naming convention:
    # - Per-league: /app/app/model-builds/{Premier League}/
    # - Fallback:   /app/app/model-builds/All Leagues/
    base = Path("/app/app/model-builds") / league_dir_name
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"{stat_name}_v{ts}.sav")


def _fit_model(X_train, y_train) -> tuple:
    """Direct newton-cholesky fit. Returns (model, algorithm_name, hyperparams_dict)."""
    model = PoissonRegressor(solver="newton-cholesky")
    model.fit(X_train, y_train)
    return model, "fit_model", {"solver": "newton-cholesky"}


def _grid_search(X_train, y_train) -> tuple:
    """CV over alpha × max_iter × fit_intercept. Returns (model, algorithm_name, best_params).

    Returns best_estimator_ only — NOT the GridSearchCV wrapper. cv_results_
    holds every fold's fitted sub-model and is the primary driver of the
    memory bloat we hit on 2026-04-22 (gunicorn OOM after ~12 grid searches).
    best_estimator_ is a PoissonRegressor; same .predict() contract so
    downstream pickle-load + .predict() at projection time is unchanged.
    """
    param_grid = {
        "alpha": np.arange(0, 1, 0.1),
        "max_iter": [100, 200, 500],
        "fit_intercept": [True, False],
    }
    pr = PoissonRegressor()
    gs = GridSearchCV(pr, param_grid, cv=5, scoring="neg_mean_squared_error")
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    best_params = dict(gs.best_params_)
    # Drop the CV wrapper + its results so they can be GC'd before next iter
    del gs
    return best, "grid_search", best_params


def _decide_promotion(incumbent: Optional[dict], new_mae: float) -> tuple:
    """Returns (should_promote, reason) — used by Phase 5; Phase 4 logs only.

    Thresholds per model_retraining_design.md open decision #6:
      - Promote if new MAE ≤ 2% worse than incumbent (allows noise)
      - Hold if 2–10% worse
      - Reject + alert if >10% worse
    Initial guesses; revisit after 3–4 dry-run cycles reveal real variance.
    """
    if incumbent is None or incumbent.get("holdout_mae") is None:
        return True, "initial (no incumbent or no baseline MAE)"

    incumbent_mae = incumbent["holdout_mae"]
    ratio = new_mae / incumbent_mae

    if ratio <= 1.02:
        pct = (incumbent_mae - new_mae) / incumbent_mae * 100
        sign = "improved" if pct >= 0 else "within noise"
        return True, f"MAE {sign} {abs(pct):.1f}%"

    if ratio >= 1.10:
        pct = (ratio - 1) * 100
        return False, f"MAE degraded {pct:.1f}% — REJECTED (>10% worse)"

    pct = (ratio - 1) * 100
    return False, f"MAE slightly worse ({pct:.1f}%) — holding"


async def _train_one(
    df: pd.DataFrame,
    stat_name: str,
    competition_id: Optional[int],
    label: str,
) -> Optional[dict]:
    """Train one (optionally scoped) model. Returns summary dict or None if
    insufficient data / training fails."""
    predictors = _predictor_columns(stat_name)
    target = _target_column(stat_name)

    needed = list(predictors) + [target]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.warning(f"[retrain {label}] {stat_name}: missing columns {missing} — skipping")
        return None

    train_df = df.dropna(subset=needed)
    if len(train_df) < MIN_TRAINING_ROWS:
        logger.info(
            f"[retrain {label}] {stat_name}: only {len(train_df)} rows "
            f"(< {MIN_TRAINING_ROWS}) — skipping"
        )
        return None

    X = train_df[list(predictors)]
    y = train_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    n_train_rows = len(X_train)

    if stat_name in _FIT_MODEL_STATS:
        model, algo, hyperparams = _fit_model(X_train, y_train)
    else:
        model, algo, hyperparams = _grid_search(X_train, y_train)

    y_pred = model.predict(X_test)
    holdout_mae = float(mean_absolute_error(y_test, y_pred))
    holdout_r2 = float(r2_score(y_test, y_pred))

    # Save .sav
    league_dir = label if competition_id is not None else "All Leagues"
    file_path = _model_build_path(league_dir, stat_name)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    new_id = await insert_projection_model(
        competition_id=competition_id,
        stat_name=stat_name,
        algorithm=algo,
        hyperparams=hyperparams,
        trained_on_n_rows=n_train_rows,
        holdout_mae=holdout_mae,
        holdout_r2=holdout_r2,
        file_path=file_path,
    )

    incumbent = await fetch_active_model(competition_id, stat_name)
    would_promote, reason = _decide_promotion(incumbent, holdout_mae)
    incumbent_mae = incumbent.get("holdout_mae") if incumbent else None

    # Free the bulky sklearn objects before the next iteration.
    # Previous runs OOM'd around the 12th-13th grid_search because
    # GridSearchCV retains every CV-trained sub-model in .cv_results_
    # plus all the fold predictions. gunicorn's worker grew until
    # kernel SIGKILL'd it mid-training. Explicit del + gc.collect()
    # knocks the working set back down between iterations.
    del model, X, y, X_train, X_test, y_train, y_test, y_pred, train_df
    gc.collect()

    logger.info(
        f"[retrain {label}] {stat_name}: algo={algo} rows={n_train_rows} "
        f"holdout_mae={holdout_mae:.4f} r2={holdout_r2:.4f} "
        f"incumbent_mae={incumbent_mae} would_promote={would_promote} ({reason}) "
        f"new_id={new_id}"
    )

    return {
        "stat": stat_name,
        "algorithm": algo,
        "n_rows": n_train_rows,
        "holdout_mae": holdout_mae,
        "holdout_r2": holdout_r2,
        "incumbent_mae": incumbent_mae,
        "would_promote": would_promote,
        "reason": reason,
        "new_id": new_id,
    }


async def retrain_all_models(dry_run: bool = True) -> dict:
    """Main entry point. Returns a summary dict with per-(league, stat) results.

    dry_run=True (Phase 4 default): trains + inserts projection_models rows
    with is_active=0, logs would-be-promotion decisions, does NOT flip
    is_active. Intentional — run for 3–4 cycles to calibrate thresholds.

    dry_run=False: Phase 5. Not yet supported — will call promote_model()
    after the guardrail clears.
    """
    if not dry_run:
        raise NotImplementedError("Non-dry-run retraining lands in Phase 5")

    t_start = time.time()
    logger.info(f"[retrain] START dry_run={dry_run}")

    # Pull the full dataset once; filter per-league in memory.
    all_df = await load_model_dataset_async()
    logger.info(f"[retrain] loaded {len(all_df)} total rows from projection_model_dataset")

    stat_list = [s for s in get_stat_list() if s != "Goals"]
    results = {"per_league": {}, "all_leagues": {}, "skipped_leagues": []}

    # Per-league models
    for league_id in sorted(all_df["comp_id"].dropna().unique().tolist()):
        league_df = all_df[all_df["comp_id"] == league_id]
        league_name = str(league_id)  # placeholder; looked up below
        try:
            league_name = await _lookup_league_name(int(league_id))
        except Exception as e:
            logger.warning(f"[retrain] couldn't resolve name for comp_id={league_id}: {e}")

        if len(league_df) < MIN_TRAINING_ROWS:
            logger.info(
                f"[retrain {league_name}] only {len(league_df)} rows — skipping (fallback to All Leagues)"
            )
            results["skipped_leagues"].append({"league_id": int(league_id), "league": league_name, "rows": len(league_df)})
            continue

        league_results = []
        for stat in stat_list:
            r = await _train_one(league_df, stat, int(league_id), league_name)
            if r is not None:
                league_results.append(r)
        results["per_league"][league_name] = league_results

    # All Leagues fallback — train on the whole pool, competition_id=NULL
    logger.info(f"[retrain All Leagues] training {len(all_df)} rows across all leagues")
    all_results = []
    for stat in stat_list:
        r = await _train_one(all_df, stat, None, "All Leagues")
        if r is not None:
            all_results.append(r)
    results["all_leagues"] = all_results

    elapsed = (time.time() - t_start) / 60
    logger.info(
        f"[retrain] COMPLETE dry_run={dry_run} — {elapsed:.1f} min, "
        f"{sum(len(v) for v in results['per_league'].values())} per-league models + "
        f"{len(all_results)} All Leagues models"
    )
    return results


async def _lookup_league_name(competition_id: int) -> str:
    """Resolve competition_id → name via the competitions table."""
    import app.database as _db
    from app.database import get_connection
    conn = None
    try:
        conn = await get_connection()
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT name FROM competitions WHERE id = %s", (competition_id,))
            row = await cursor.fetchone()
            if not row:
                raise ValueError(f"no competition with id={competition_id}")
            return row[0]
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)
