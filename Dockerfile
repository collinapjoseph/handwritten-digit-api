# Base image with lightweight python installed
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy and install dependencies first, before copying application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files.
COPY server.py .
COPY mnist_cnn.onnx .
COPY mnist_cnn.onnx.data .
COPY ui.html .

# Document which port the container listens on.
# EXPOSE is metadata only — it does not publish the port.
# The actual binding is done at runtime 
EXPOSE 8000

# Start the server. 
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
