.PHONY: setup preprocess clean fetch

setup:
	pip install -U pip && .\.venv\Scripts\python -m pip install -r requirements.txt

preprocess:
	python scripts/preprocess.py --config configs/default.yaml

clean:
	rm -rf data/processed/* reports/*

fetch:
	python scripts/fetch_kaggle.py
