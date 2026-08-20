# Use a secure, official Python slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (essential for building python packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app.py retrieval_test.py ./
COPY src/ ./src/
COPY data/ ./data/

# Expose FastAPI server port
EXPOSE 8000

# Run the clinical hepatology RAG server, binding to 0.0.0.0
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]
