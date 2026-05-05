FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install -e . --no-cache-dir

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the API server
CMD ["python", "-m", "uvicorn", "blend.api:app", "--host", "0.0.0.0", "--port", "8080"]
