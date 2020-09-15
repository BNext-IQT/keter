import os
from time import sleep
from datetime import timedelta
from pathlib import Path
from rq import Queue, Worker, Connection
from redis import Redis

_REDIS_HOST = os.environ.get('KETER_QUEUE') or ''
CPU = Queue(name='cpu', connection=Redis(_REDIS_HOST))
GPU = Queue(name='gpu', connection=Redis(_REDIS_HOST))

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

def work(queue):
    if queue == 'all':
        queue = ['cpu', 'gpu']
    with Connection(Redis(_REDIS_HOST)):
        worker = Worker(queue)
        worker.work(with_scheduler=True)

def up():
    if not _chembl_data_exists():
        CPU.enqueue(download_chembl)
    if not _nyt_data_exists():
        CPU.enqueue(download_nyt)
    if not _chemistry_model_exists():
        GPU.enqueue(chemistry_model_train)
    if not _forecast_model_exists():
        GPU.enqueue(forecast_model_train)
    CPU.enqueue(foreman)  

def foreman():
    CPU.enqueue_in(_FOREMAN_RESPAWN, foreman)
    if not _forecast_cache_is_fresh():
        GPU.enqueue(forecast_cache_infer)
        GPU.enqueue_in(_FORECAST_FRESHNESS - timedelta(hours=1), forecast_cache_infer)
    drug_discovery_jobs_to_create = len(Worker.all(queue=GPU)) * 2 - len(GPU)
    for _ in range(drug_discovery_jobs_to_create):
        GPU.enqueue(chemistry_discover_drugs)

def download_chembl():
    sleep(2)

def download_nyt():
    sleep(2)

def chemistry_model_train():
    sleep(2)

def forecast_model_train():
    sleep(2)

def forecast_cache_infer():
    sleep(2)

def chemistry_discover_drugs():
    sleep(2)