FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin ionoscope

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=ionoscope:ionoscope . .
RUN mkdir -p shadow_data production_models \
    && chown ionoscope:ionoscope /app shadow_data production_models

USER ionoscope

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
