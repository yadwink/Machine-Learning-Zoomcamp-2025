FROM python:3.13.5-slim-bookworm
WORKDIR /code
COPY pipeline_v2.bin .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" scikit-learn==1.6.1
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]