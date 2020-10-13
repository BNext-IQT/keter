import os
from time import sleep
from datetime import timedelta
from pathlib import Path
from rq import Queue, Worker, Connection
from redis import Redis
import redis.exceptions
from sqlalchemy import create_engine

_CACHE = Path(os.environ.get('KETER_CACHE') or Path.home() / '.keter')
_CACHE.mkdir(parents=True, exist_ok=True)

_FORECAST_FRESHNESS = timedelta(hours=24)
_FOREMAN_RESPAWN = timedelta(minutes=30)

def _chembl_data_exists():
    return False

def _nyt_data_exists():
    return False

def _chemistry_model_exists():
    return False

def _forecast_model_exists():
    return False

def _forecast_cache_is_fresh():
    return False

def work(queue: str, redis_url: str):
    conn = Redis(redis_url)
    if queue == 'all':
        queue = ['cpu', 'gpu']
    with Connection(conn):
        worker = Worker(queue)
        worker.work(with_scheduler=True)

def foreman(redis_url: str):
    conn = Redis(redis_url)
    foreman_respawn_time = timedelta(minutes=30)

    cpu = Queue(name='cpu', connection=conn)
    gpu = Queue(name='gpu', connection=conn)

    cpu.enqueue_in(foreman_respawn_time, foreman, redis_url)

    if not _chemistry_model_exists():
        gpu.enqueue(chemistry_model_train)
    if not _forecast_model_exists():
        gpu.enqueue(forecast_model_train)  
    if not _forecast_cache_is_fresh():
        cpu.enqueue(coronavirus_cases_update)
        gpu.enqueue(forecast_cache_infer)
    drug_discovery_jobs_to_create = len(Worker.all(queue=gpu)) * 2 - len(gpu)
    for _ in range(drug_discovery_jobs_to_create):
        gpu.enqueue(chemistry_discover_drugs)

def coronavirus_cases_update():
    sleep(2)

def chemistry_model_train():
    sleep(2)

def forecast_model_train():
    sleep(2)

def forecast_cache_infer():
    sleep(2)

def chemistry_discover_drugs():
    sleep(2)

def chemistry_ingest_chembl(url):
    print(url)
    sleep(2)