# Dockerfile
FROM python:3.11

# Set work directory
WORKDIR /transformer_attentions

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the src directory
COPY src/ ./src/

# Expose the port that the application listens on.
EXPOSE 8999

# Default command
CMD ["python", "src/main.py"]