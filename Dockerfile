# TODO [SWE EXERCISE 7 — Containerisation]:
# Implement this Dockerfile. Guidelines:
#
# FROM python:3.10-slim
#
# WORKDIR /app
#
# # Install dependencies first (layer caching — deps change less often than code)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
#
# # Copy source code
# COPY . .
#
# # Expose ports for API and dashboard
# EXPOSE 8000 8501
#
# # Default command (can be overridden in docker-compose)
# CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
#
# LEARN: Layer caching (COPY requirements.txt before COPY . .), why slim images,
# EXPOSE vs port binding, CMD vs ENTRYPOINT.

# PLACEHOLDER — replace this entire file with the real Dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN echo "TODO: Implement this Dockerfile (SWE Exercise 7)"
