FROM python:3.11-slim

WORKDIR /backend/

COPY requirements.txt .
COPY . .

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]