# Use an official Python runtime as a parent image
# Using 3.14 as requested (if available), otherwise fallback to 3.13 or 3.12
FROM python:3.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# libgomp1 is required for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Create directories for data persistence
RUN mkdir -p data models logs

# Expose the port for the dashboard
EXPOSE 8787

# Default command (can be overridden by docker-compose)
CMD ["bash"]
