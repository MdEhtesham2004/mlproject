# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app/src
# Copy the current directory contents into the container at /app
COPY . /app/src

# Install dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the application to be accessed
EXPOSE 5000

# Define environment variable for production
ENV FLASK_ENV=production

# Command to run the application using Gunicorn with 4 workers
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:5000", " src/app:app"]
