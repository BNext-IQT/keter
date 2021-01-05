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
from keter.chemistry import (
    GraphConvModelAnalyzer,
    get_data,
    save_corpus,
    read_corpus,
    transform_elemental,
)
from keter.config import *


def _queue_jobs(queue: str, job: Callable, repeats=1):
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
    foreman_respawn_time = timedelta(minutes=90)

    cpu = Queue(name="cpu", connection=conn, default_timeout=7200)
    gpu = Queue(name="gpu", connection=conn, default_timeout=36000)
    cpu.enqueue_in(foreman_respawn_time, foreman)

    if not CACHE_MOLS.exists():
        cpu.enqueue(create_dataset_and_transformations)

    cpu.enqueue(coronavirus_cases_update)
    gpu.enqueue(chemistry_model_train)
    gpu.enqueue(forecast_model_train)


def coronavirus_cases_update():
    sleep(2)


def chemistry_rage_and_serenity_train():
    corpus = read_corpus(CACHE_FEATURES_ELE_LANG)
    serenity = Serenity()
    serenity.fit(corpus, CACHE_MODELS)


def chemistry_gcn_train():
    chem = GraphConvModelAnalyzer(CACHE_MODELS)
    chem.fit()

    _queue_jobs("gpu", chemistry_infer_drugs, DRUG_DISCOVERY_JOBS_PER_MODEL)


def forecast_model_train():
    _queue_jobs(FORECASTING_JOBS_PER_MODEL, "gpu", forecast_cache_infer)


def forecast_cache_infer():
    sleep(2)


def chemistry_infer_drugs():
    sleep(2)
    _queue_jobs("cpu", chemistry_sift_for_drugs)


def chemistry_sift_for_drugs():
    sleep(2)


def create_dataset_and_transformations():
    dataset = get_data()
    save_corpus(str(CACHE_FLAIR_CORPUS), transform_elemental(dataset))
    dataset.to_parquet(CACHE_MOLS)
