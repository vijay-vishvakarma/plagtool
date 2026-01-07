# Use official Python slim image (lightweight, based on Debian)
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Expose Streamlit port
EXPOSE 8501

# Install system dependencies (for spacy and textstat)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements (including spacy model)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data if needed (but we're NLTK-free, so optional)
# RUN python -m nltk.downloader punkt averaged_perceptron_tagger wordnet

# Copy your app code
COPY . .

# Health check (optional)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit app
ENTRYPOINT ["streamlit", "run"]
CMD ["rewritter.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
