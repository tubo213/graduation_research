#!/bin/sh
for i in {126..234};
do
    echo "Running experiment optimize$i"
    poetry run python src/bin/optimize.py --exp optimize$i
done
