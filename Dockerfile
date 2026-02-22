FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[demo,test]"

EXPOSE 8501

CMD ["streamlit", "run", "src/quantum_optical_bus/calibration_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
