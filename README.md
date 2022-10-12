# Mercari Coupon Allocation

## Setup

1. 以下コマンドを実行
```
git clone git@github.com:tubo213/graduation_research.git
cd graduation_research
poetry shell
poetry install
```
2. `input/`直下に[データ](https://drive.google.com/file/d/1qaAhXao5zFLmW3KMpHBa2gYXHP9FWTRb/view?usp=sharing)を置く

## Train
uplift modelを学習します.
学習の設定は`yaml/hoge.yaml`で指定します
```
poetry run python src/bin/train.py --exp test
```
`--debug`オプションをつけると小さいデータで実行します
```
poetry run python src/bin/train.py --exp test --debug true
```

## Optimize
uplift modelの予測値を用いて最適化を実行します
```
poetry run python src/bin/train.py --exp test
```
こちらも`--debug`オプションをつけると小さいデータで実行します.
trainをdebugで実行した場合にはoptimizeでも付ける必要があります
```
poetry run python src/bin/optmize.py --exp test --debug true
```

## Run
uplift modelingの学習と最適化による割当全てを実行します
```
sh bin/run.sh test false
```

## 可視化
現状notebookで結果を可視化しています.サンプル実装を以下に載せています
`notebook/evaluation/test.ipynb`
