FROM python:3.9

# Set working directory to /code
WORKDIR /code

# Copy specific requirement file
COPY backend/requirements_web.txt /code/requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the entire repository
COPY . /code

# Create necessary directories that the app writes to
RUN mkdir -p /code/backend/uploads \
    /code/backend/history_uploads \
    /code/backend/feedback_images \
    /code/model/results/checkpoints

# Set permissions for writable directories (required for Spaces running as non-root)
RUN chmod -R 777 /code/backend/uploads \
    /code/backend/history_uploads \
    /code/backend/feedback_images

# Ensure database exists or is writable
RUN touch /code/backend/database.db && chmod 777 /code/backend/database.db

# Expose the port
EXPOSE 7860

# Run the backend app
CMD ["python", "backend/app.py"]
