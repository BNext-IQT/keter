import os
from time import sleep
from datetime import timedelta
from pathlib import Path
from rq import Queue, Worker, Connection
from redis import Redis
import redis.exceptions
import pandas as pd
from sqlalchemy import create_engine
from keter.data import get_smiles_from_chembl
from keter.chemistry import Chemistry

CACHE = Path(os.environ.get('KETER_CACHE') or Path.home() / '.keter')
(CACHE / 'data' / 'molnet').mkdir(parents=True, exist_ok=True)
QUEUE = os.environ.get('KETER_QUEUE') or ''

def load_df(name: str) -> pd.DataFrame:
    df_file = (CACHE / name).with_suffix('.df')
    if not df_file.exists():
        print("Cloud not implemented yet")
        exit(-1)
    
    return pd.read_parquet(df_file)

def dump_df(name: str, df: pd.DataFrame):
    df_file = (CACHE / name).with_suffix('.df')
    df.to_parquet(df_file)

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

def work(queue: str):
    conn = Redis(QUEUE)
    if queue == 'all':
        queue = ['cpu', 'gpu']
    with Connection(conn):
        worker = Worker(queue)
        worker.work(with_scheduler=True)

def foreman():
    conn = Redis(QUEUE)
    foreman_respawn_time = timedelta(minutes=30)

    cpu = Queue(name='cpu', connection=conn)
    gpu = Queue(name='gpu', connection=conn)

    cpu.enqueue_in(foreman_respawn_time, foreman)

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
    chem = Chemistry(CACHE)
    chem.fit()
    print(chem.score())
    

def forecast_model_train():
    sleep(2)

def forecast_cache_infer():
    sleep(2)

def chemistry_discover_drugs():
    sleep(2)

def chemistry_ingest_chembl(dburl):
    conn = create_engine(dburl)
    df = get_smiles_from_chembl(conn)
    dump_df('smiles', df)