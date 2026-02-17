.PHONY: test lint app

test:
	python -m pytest -q

lint:
	ruff format --check .
	ruff check .
	python -m compileall src tests

app:
	streamlit run src/quantum_optical_bus/calibration_app.py
