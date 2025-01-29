FROM python:3.10
WORKDIR /app/src
COPY . /app/src
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000

CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:5000", " app:app"]
