## Fine-tuning LLMs for Coreference Resolution

### Overview

This is to fine-tune open LLMs (e.g., `Gemma` family) using [`mlx-lm`](https://github.com/ml-explore/mlx-lm) for Coreference Resolution.

#### Coreference Resolution 

**Coreference Resolution** is to restore the entities in the text which was modified to prevent the repetitions and to ensure linguistic varieties. 

For instance, the following text becomes

> Paul went to his local Costco. He made sure to buy six packs.

Paul went to Paul's local Costco. Paul made sure to buy six packs. 

Coreference Resolution will help the computers to perform text processing, to avoid the ambiguity and to aid the better text analysis, e.g., Sentiment Analysis.


### Requirements

Needless to say, since [`mlx-lm`](https://github.com/ml-explore/mlx-lm) is developed and optimized for the Apple Silicon devices (e.g., Macbook Air M1), the whole code in this repo is only applicable for Apple Silicon.

Additionally, it is assumed that [`uv`](https://docs.astral.sh/uv/) has been installed.


### File structures

#### Training Preparation

The following files are used as prerequisite data / configurations.

* `training_config.yml`: The training configuration is defined in this `yaml` file
* `data/coreference_dataset.csv`: The original dataset to be used in the training (`original`: Original Text, `coref`: Desired text)

To prepare the training, run the `coref_prep.py` file to generate the data in the `json` format which is accepted by [`mlx-lm`](https://github.com/ml-explore/mlx).

Usage:

```
uv run python coref_prep.py
```

This will create the `training.jsonl` and `valid.jsonl` datasets within the `data\` directory.


#### Training

To start the training (fine-tuning), run the `coref_training.py` script with datasets and configuration (`training_config.yml`) files in hand.

Usage:

```
uv run python coref_training.py
```

