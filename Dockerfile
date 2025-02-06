FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt from the root directory to /app/
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy the rest of the project files
COPY ProjectA2/ /app/

# Expose the Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]