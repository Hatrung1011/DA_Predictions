FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for lightgbm and xlsxwriter
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the script
CMD ["python", "main.py"]
