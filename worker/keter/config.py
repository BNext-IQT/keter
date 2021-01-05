import os
from pathlib import Path

# Configuration for default locations for cache files and folders
CACHE_ROOT = Path(os.environ.get("KETER_CACHE") or Path.home() / ".keter")
CACHE_DATASET = CACHE_ROOT / "dataset"
CACHE_MODELS = CACHE_ROOT / "models"
CACHE_MOLS = CACHE_DATASET / "original.parquet"
CACHE_FLAIR_CORPUS = CACHE_DATASET / "elemental_language.pickle.xz"

# The Redis hostname for storing global data and getting jobs
QUEUE = os.environ.get("KETER_QUEUE") or ""

# Default number of jobs to spawn when we train a new model.
DRUG_DISCOVERY_JOBS_PER_MODEL = 10
FORECASTING_JOBS_PER_MODEL = 10

# Unit of work qualitatively controls how long jobs that rely on iteration should last.
# The higher this is, the less Redis traffic but more expensive job failures are.
# Realistically the system can scale to tens of thousands of workers on the default setting because Redis is very fast.
UNIT_OF_WORK = "medium"

# How the cache should behave in systems which use it.
# "required" = All files must pre-cached (models, datasets) when loading a system or an Exception is thrown.
# "auto" = If a file is missing or stale, the system tries to create it de novo.
#          (like training a new model if the model file is missing, or downloading a dataset if missing).
# "build" = Rebuilds the cache no matter what. Very slow.
CACHE_BEHAVIOR = "auto"

# Create directories if they don't exist.
CACHE_DATASET.mkdir(parents=True, exist_ok=True)
CACHE_MODELS.mkdir(exist_ok=True)
