# Use the base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /workspace

# Copy the project files to the Docker image
COPY . /workspace
RUN rm -rf /workspace/out
RUN rm -rf /workspace/venv
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
RUN find . | grep -E "(\.egg-info$)" | xargs rm -rf

# Activate the virtual environment and install dependencies
RUN /bin/bash -c "pip install --upgrade pip && \
    pip install -r requirements.txt"

# Set a default port environment variable
ENV PORT=5019

# Set the HF_HOME environment variable to /workspace/data
ENV HF_HOME=/workspace/data

# Expose the port
EXPOSE ${PORT}

# Command to run the application
CMD ["python", "app.py", "--port", "5019"]