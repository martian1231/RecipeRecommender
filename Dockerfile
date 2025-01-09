# Use the official Python image as a base
FROM python:3.9-slim

# Update apt package list
RUN apt update

# Install unzip utility to extract files
RUN apt install unzip

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container's /app directory
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose port 9090 to allow the Streamlit app to be accessed
EXPOSE 9090

# Command to run the Streamlit app on startup
CMD ["streamlit", "run", "app.py", "--server.port=9090", "--server.address=0.0.0.0"]
