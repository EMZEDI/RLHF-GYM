# Use continuumio/miniconda3 as the base image
FROM continuumio/miniconda3

# Create the working directory and set PYTHONPATH
WORKDIR /app
ENV PYTHONPATH=/app

# Copy requirements.txt (if you have one)
COPY requirements.txt ./

# Create a Python 3.11 environment and activate it as default
RUN conda create -n myenv python=3.11  \
    && echo "conda activate myenv" >> ~/.bashrc

# Install dependencies (replace with your actual package names)
RUN conda install --name myenv --file requirements.txt  # Use conda for package management

# Expose port 80
EXPOSE 80

# Start a bash shell in the /app directory
CMD ["bash"]