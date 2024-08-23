# Use the official Python image as the base
FROM python:3.9.19

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port your application listens on (if applicable)
# EXPOSE 8000

# Set the command to run your application
CMD ["python", "app.py"]
