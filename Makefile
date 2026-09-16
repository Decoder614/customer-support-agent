.PHONY: test prepare train evaluate demo reproduce clean

test:
	python -m pytest

prepare:
	python -m src.data.loader

train:
	python -m src.intents.train

evaluate:
	python -m src.evaluation.metrics

demo:
	python -m src.agent.support_agent --message "I left my wallet in the car"

reproduce:
	python scripts/reproduce_results.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').glob('.pytest_cache')]"
