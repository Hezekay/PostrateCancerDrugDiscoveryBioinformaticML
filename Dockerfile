# Use Python 3.11 slim as base
FROM python:3.11-slim

# Install system dependencies (Java + libgomp)
RUN apt-get update && apt-get install -y default-jre libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# ✅ Ensure PaDEL-Descriptor folder is copied correctly
COPY PaDEL-Descriptor /app/PaDEL-Descriptor

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# ✅ Use Gunicorn for production (removes Flask warning)
CMD ["gunicorn", "--timeout", "300", "--workers", "1", "--threads", "2", "--log-level", "debug", "--bind", "0.0.0.0:5000", "app:app"]

