FROM python:3.9-slim

WORKDIR /model
COPY movement-classification /model

EXPOSE 80
CMD ["python3", "-m", "http.server", "80"]
