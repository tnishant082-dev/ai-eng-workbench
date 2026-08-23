"""Scheduled job entrypoint — APScheduler if available, else CLI docs for cron."""
from __future__ import annotations

import argparse
from typing import Callable


def run_scheduled(job: Callable[[], None], seconds: int = 3600) -> str:
    """
    If APScheduler installed, start a BackgroundScheduler.
    Otherwise print cron guidance and run once.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler

        sched = BlockingScheduler()
        sched.add_job(job, "interval", seconds=seconds, id="aiwb_job")
        print(f"APScheduler running every {seconds}s — Ctrl+C to stop")
        sched.start()
        return "apscheduler"
    except Exception:
        print(
            "APScheduler not installed. Run once now, and schedule via cron, e.g.:\n"
            f"  */60 * * * *  cd /path/to/repo && python -m app.dataeng.scheduler --once\n"
        )
        job()
        return "cron_docs_once"


def main() -> None:
    parser = argparse.ArgumentParser(description="AIWB data eng scheduled entrypoint")
    parser.add_argument("--once", action="store_true", help="Run ETL once and exit")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between runs (APScheduler)")
    args = parser.parse_args()

    def job() -> None:
        from app.core.config import get_settings
        from app.dataeng.etl import run_etl
        from app.dataeng.quality import corpus_quality, tabular_quality
        from app.ml.pipeline import FEATURE_COLS, TARGET

        s = get_settings()
        etl = run_etl(s.corpus_dir, s.chunk_size, s.chunk_overlap, s.catalog_path, s.lineage_path)
        cq = corpus_quality(s.corpus_dir)
        tq = tabular_quality(s.dataset_path, FEATURE_COLS + [TARGET])
        print({"etl_chunks": etl.get("chunks"), "corpus_quality": cq, "tabular_quality": tq})

    if args.once:
        job()
    else:
        run_scheduled(job, seconds=args.interval)


if __name__ == "__main__":
    main()
