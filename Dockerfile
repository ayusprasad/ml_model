FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-app.txt .
RUN python -m pip install --no-cache-dir -r requirements-app.txt
COPY app.py .
EXPOSE 5000
CMD ["python", "app.py"]
