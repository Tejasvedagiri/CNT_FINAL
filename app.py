import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Generate some random data
input_data = torch.randn(128, 3, 224, 224).to(device)

# Define your model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.to(device)

# Run a forward pass and measure the time
start = time.time()
with torch.no_grad():
    output = model(input_data)
end = time.time()

print('Elapsed time:', end - start)