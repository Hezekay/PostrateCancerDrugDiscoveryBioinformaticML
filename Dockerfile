# Using Python 3.11 slim as base
FROM python:3.11-slim

# Installing system dependencies (Java + libgomp)
RUN apt-get update && apt-get install -y default-jre libgomp1 && rm -rf /var/lib/apt/lists/*

# Setting working directory
WORKDIR /app

# Copying all project files
COPY . /app

# This ensure that PaDEL-Descriptor folder is copied correctly
COPY PaDEL-Descriptor /app/PaDEL-Descriptor

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Exposing Flask port
EXPOSE 5000

# Using Gunicorn for production (removes Flask warning)
CMD ["gunicorn", "--timeout", "300", "--workers", "1", "--threads", "2", "--log-level", "debug", "--bind", "0.0.0.0:5000", "app:app"]

