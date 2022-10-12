#!/bin/sh
poetry run python src/bin/train.py --exp $1 --debug $2
poetry run python src/bin/optimize.py --exp $1 --debug $2
