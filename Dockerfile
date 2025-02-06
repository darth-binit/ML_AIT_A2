# Use the official Python image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy project files into the container
COPY ProjectA2/ /app/

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]