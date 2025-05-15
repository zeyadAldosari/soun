FROM python:3.13.3-slim

WORKDIR /app/backend

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    dos2unix \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN dos2unix /entrypoint.sh

CMD ["python",  "app.py"]