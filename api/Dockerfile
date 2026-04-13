FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy API code and model module
COPY api/app.py .
COPY dqn_agent_pytorch.py .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "app.py"]