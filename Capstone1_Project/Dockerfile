# Dockerfile

# 1. Base image (use a stable Python version)
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install system dependencies (optional but often useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the project
COPY . .

# 7. Expose the FastAPI port
EXPOSE 8000

# 8. Start the API with Uvicorn
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
