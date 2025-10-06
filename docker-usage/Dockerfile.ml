# ML Training Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install ML dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    matplotlib \
    seaborn \
    jupyter \
    psycopg2-binary

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 ml_user && chown -R ml_user:ml_user /app
USER ml_user

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
