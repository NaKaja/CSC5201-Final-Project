FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt
EXPOSE 5001
CMD ["python", "model_service.py"]