FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY sarvam_flash.py sarvam-v4-voices.json ./
ENV PORT=8765
EXPOSE 8765
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --ws-ping-interval 20"]
