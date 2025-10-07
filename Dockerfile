# 1️⃣ Use Python 3.11 (since you’re using that version)
FROM python:3.11-slim

# 2️⃣ Install Java (needed for descriptor generation)
RUN apt-get update && apt-get install -y default-jre

# 3️⃣ Set working directory
WORKDIR /app

# 4️⃣ Copy all files from your project
COPY . .

# 5️⃣ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Expose port 5000 (Flask default)
EXPOSE 5000

# 7️⃣ Start your Flask app
CMD ["python", "app.py"]
