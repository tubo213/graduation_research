#!/bin/sh
poetry run python src/bin/train.py --exp $1 --debug $3
poetry run python src/bin/optimize.py --exp $2 --debug $3
