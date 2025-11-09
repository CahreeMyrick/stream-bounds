# Makefile
.PHONY: install run test lint
install:
\tpython -m pip install -e . pytest

run:
\tpython -m src.main --n 200000 --seed 7 --outlier-rate 0.005 --q 0.1 --q 0.5 --q 0.9

test:
\tpython -m pytest -q

lint:
\tpython -m pip install ruff && ruff check src tests

