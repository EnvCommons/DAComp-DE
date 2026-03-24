FROM python:3.11-slim

RUN apt update && apt upgrade -y && apt install -y curl git

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY dacomp_de.py .
COPY evaluate_de.py .

EXPOSE 8080

CMD ["python", "server.py"]
