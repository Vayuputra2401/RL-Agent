FROM public.ecr.aws/docker/library/python:3.11-slim

# HF Spaces listens on port 7860
ENV PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/        ./app/
COPY openenv.yaml .
COPY inference.py .

EXPOSE 7860

RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
