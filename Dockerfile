# Start from NVIDIA’s PyTorch container
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Set workdir
WORKDIR /workspace

# Copy requirements first and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Default command
CMD ["/bin/bash"]
