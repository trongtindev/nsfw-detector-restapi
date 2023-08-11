FROM python:3.9
RUN pip install nsfw-detector
COPY ./ ./
RUN pip install -r requirements.txt
CMD ["python", "src/main.py"]