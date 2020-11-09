import os
from typing import Callable
from time import sleep
from datetime import timedelta
from pathlib import Path
from rq import Queue, Worker, Connection
from redis import Redis
import redis.exceptions
import pandas as pd
from sqlalchemy import create_engine
from keter.data import gather_mols_with_props, transform_elemental_language
from keter.chemistry import Chemistry

CACHE_ROOT = Path(os.environ.get("KETER_CACHE") or Path.home() / ".keter")
CACHE_GROUND_TRUTH = CACHE_ROOT / "ground_truth"
CACHE_FEATURES = CACHE_ROOT / "features"
CACHE_MODELS = CACHE_ROOT / "models"
CACHE_MOLS = CACHE_GROUND_TRUTH / "mols.parquet"
CACHE_FEATURES_ELE_LANG = CACHE_FEATURES / "elemental.txt.gz"

CACHE_GROUND_TRUTH.mkdir(parents=True, exist_ok=True)
CACHE_FEATURES.mkdir(exist_ok=True)
CACHE_MODELS.mkdir(exist_ok=True)

DRUG_DISCOVERY_JOBS_PER_MODEL = 10
FORECASTING_JOBS_PER_MODEL = 10

QUEUE = os.environ.get("KETER_QUEUE") or ""


def _queue_repeating_jobs(repeats: int, queue: str, job: Callable):
    conn = Redis(QUEUE)
    job_queue = Queue(name=queue, connection=conn, default_timeout=36000)
    for _ in range(repeats):
        job_queue.enqueue(job)


def work(queue: str):
    conn = Redis(QUEUE)
    if queue == "all":
        queue = ["cpu", "gpu"]
    with Connection(conn):
        worker = Worker(queue)
        worker.work(with_scheduler=True)


def foreman():
    conn = Redis(QUEUE)
    foreman_respawn_time = timedelta(minutes=5)

    cpu = Queue(name="cpu", connection=conn, default_timeout=7200)
    gpu = Queue(name="gpu", connection=conn, default_timeout=36000)
    cpu.enqueue_in(foreman_respawn_time, foreman)

    if not CACHE_MOLS.exists():
        cpu.enqueue(create_datasets)
    else:
        if not CACHE_FEATURES_ELE_LANG.exists():
            cpu.enqueue(create_elemental_language)

    cpu.enqueue(coronavirus_cases_update)
    gpu.enqueue(chemistry_model_train)
    gpu.enqueue(forecast_model_train)


def coronavirus_cases_update():
    sleep(2)


def chemistry_model_train():
    chem = Chemistry(CACHE_MODELS)
    chem.fit()
    print(chem.score())

    _queue_repeating_jobs(
        DRUG_DISCOVERY_JOBS_PER_MODEL, "gpu", chemistry_discover_drugs
    )


def forecast_model_train():
    _queue_repeating_jobs(FORECASTING_JOBS_PER_MODEL, "gpu", forecast_cache_infer)


def forecast_cache_infer():
    sleep(2)


def chemistry_discover_drugs():
    sleep(2)


def create_datasets():
    gather_mols_with_props().to_parquet(CACHE_MOLS)


def create_elemental_language():
    dataset = pd.read_parquet(CACHE_MOLS)
    transform_elemental_language(dataset, str(CACHE_FEATURES_ELE_LANG))
