.PHONY: setup pipeline train serve test monitor lint

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

pipeline:
	dvc repro

train:
	python src/train/train.py --params params.yaml

serve:
	docker compose up --build

test:
	pytest tests/ -v

monitor:
	python src/monitoring/monitor.py \
		--reference data/processed/test.csv \
		--current data/live/predictions.csv \
		--params params.yaml

lint:
	flake8 src --max-line-length=100 --extend-ignore=E203
	black src
