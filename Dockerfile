FROM python:3.9-alpine

WORKDIR /python-docker

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN apk add --no-cache gcc musl-dev libffi-dev && \
    pip3 install --no-cache-dir -r requirements.txt && \
    apk del gcc musl-dev libffi-dev

# Copy application code
COPY . .

# Create a new user with UID 10016
RUN addgroup -g 10016 choreo && \
    adduser --disabled-password --no-create-home --uid 10016 --ingroup choreo choreouser

USER 10016

# Expose port and set command
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]
