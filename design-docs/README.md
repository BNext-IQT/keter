Keter is a health intelligence platform for addressing a pandemic with an eye on artificial intelligent solutions. Right now it focuses on small molecule antivirals and forecasting. It may move in other directions as new ideas come.

This document describes the architecture and design of the Keter system. Keter is still a concept project and things may change suddenly.

# Machine learning architecture
This document describes the common machine learning architecture used in Keter.

#### Models
Machine learning models that take in raw processed data. Models need not include any preprocessors needed to create the tensor or other data structures actually used by the model, but can implement them as separate ``Transformer`` layers. Models need not be polymorphic, although should expose a similar interface if reasonably possible. Models should contain no code that commits to the cache, and any codepaths that involve writing to files must be configurable.

#### Actors
Actors are modules that include one model, a dataset it is trained on, and one or more hyperparamter sets. Actors should be implemented using the Ray actor toolkit. Actors are allowed to commit to the cache.

#### Systems
Systems are modules that include one or more actors. The output of a system is always the product in question, which is implemented as a generator of a Protocol Buffer (``DrugCandidate``, ``Forecasts``). For example, a system for drug discovery should implement a ``iterdrugs`` method which returns a generator for ``DrugCandidate``s. Systems should not commit to the cache directly, but Actors they include may do so commit, so using a System may modify the cache.

#### Datasets
Datasets should download automatically from the Internet and commit themselves to the cache. When queried for data, a Dataset return a Pandas DataFrame. A dataset must be stored in the cache in the Parquet format using the ``to_parquet`` method of Pandas. Reading the DataFrame uses the ``read_parquet`` of Pandas. Do not store DataFrames as CSV, even if it was the original source format! 

The API is like this. To get a DataFrame from a Dataset, you simply call it:

```python
mols = Molecules()
dataframe = mols()
# This fails if the dataset hasn't been downloaded already.
dataframe = mols(cache='required')
```
#### Cache
The cache is a Git repository who's location is set the environment variable ``$KETER_CACHE`` or ``~/.keter`` if not specified. DataFrames, Model files, and other objects are stored in this repository.

We are currently investigating [Data Version Control](https://dvc.org/).

#### Driver
A Driver uses a system to produce a Product, which can be a Report or a changeset for a Database.

