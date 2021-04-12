Keter is a health intelligence platform for addressing a pandemic with an eye on artificial intelligent solutions. Right now it focuses on small molecule antivirals and forecasting. It may move in other directions as new ideas come.

This document describes the architecture and design of the Keter system. Keter is still a concept project and things may change suddenly.

# Machine learning architecture
This document describes the common machine learning architecture used in Keter.

#### Models
Machine learning models that take in raw processed data. Models need not include any preprocessors needed to create the tensor or other data structures actually used by the model, but can implement them as separate ``Transformer`` layers. Models need not be polymorphic, although should expose a similar interface if reasonably possible. Models should contain no code that commits to the cache, and any codepaths that involve writing to files must be configurable.

#### Datasets
Datasets should download automatically from the Internet and commit themselves to the cache. When queried for data, a Dataset return a Pandas DataFrame. A dataset must be stored in the cache in the Parquet format using the ``to_parquet`` method of Pandas. Reading the DataFrame uses the ``read_parquet`` of Pandas. Do not store DataFrames as CSV, even if it was the original source format! 

The API is like this. To get a DataFrame from a Dataset, you simply call it:

```python
tox = Toxicity()
dataframe = tox.to_df()
```
#### Actors
Actors are modules that include one model, a dataset it is trained on, and one or more hyperparamter sets. Actors require a stage to "act" on. Actors may have modes that change their behavior.

#### Stage
Stages control how and if data is persisted. This will be a more complicated subsystem eventually, but right now it's simple and just stores files in the file system.

We are currently investigating [Data Version Control](https://dvc.org/) and the [Ray](https://ray.io/) for their features for creating actors and stages.

#### Productions
Productions are functions that include datasets and actors operating on a stage.


