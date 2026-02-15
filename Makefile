.PHONY: test lint app

test:
	python -m pytest -q

lint:
	python -m compileall src tests

app:
	streamlit run src/quantum_optical_bus/calibration_app.py
