# Use Python 3.11 slim as base
FROM python:3.11-slim

# Install system dependencies (Java + libgomp)
RUN apt-get update && apt-get install -y default-jre libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run your Flask app
CMD ["python", "app.py"]
