.PHONY: setup preprocess clean fetch baselines

setup:
	pip install -U pip && .\.venv\Scripts\python -m pip install -r requirements.txt

preprocess:
	python scripts/preprocess.py --config configs/default.yaml

clean:
	rm -rf data/processed/* reports/*

fetch:
	python scripts/fetch_kaggle.py

baselines:
	python models/baselines.py --meta data/processed/meta.json --reports_dir reports --config configs/default.yaml
