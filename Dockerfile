FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps from requirements.txt, explicitly including the new lint/CI tools.
COPY requirements.txt ./
RUN pip install "torch>=2.2,<2.8" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . /app

CMD ["python", "-m", "src.training.train"]
