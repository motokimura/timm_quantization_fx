## Usage

### Prepare ImageNet dataset

Download ImageNet dataset under `$HOME/data/imagenet`.

```
$ tree $HOME/data/imagenet -L 1

/home/motoki_kimura/data/imagenet
├── test
├── train
└── val
```

You may use [kaggle /imagenet-object-localization-challenge dataset](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
to download ImageNet dataset.

`test` set is not used in this repository. Use [scripts/valprep.sh](scripts/valprep.sh) if you need to preprocess `val` set.

## Prepare Docker container

```
$ docker compose run --rm dev bash
```
