# Base image with Python
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Set up working directory
WORKDIR /app

# Install dependencies
COPY conda.yml .
RUN conda env create -f conda.yml
SHELL ["conda", "run", "-n", "mcmc-env", "/bin/bash", "-c"]

# Copy your script and other files
COPY . .

# Set default command
CMD ["python", "run_mh_parallel.py"]