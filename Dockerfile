FROM python:3.9-alpine

WORKDIR /python-docker

# Install system dependencies
RUN apk add --no-cache gcc musl-dev libffi-dev mesa-gl

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a new user with UID 10016
RUN addgroup -g 10016 choreo && \
    adduser --disabled-password --no-create-home --uid 10016 --ingroup choreo choreouser

USER 10016

# Expose port and set command
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]
