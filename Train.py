import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import signal
import sys
from DataLoader import get_batch, hyperparameters
from Transformer import MyModel
import torch.optim.lr_scheduler as lr_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training.log', filemode='w')

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda_available = torch.cuda.is_available()
logging.info(f"CUDA available: {cuda_available}")

if cuda_available:
    logging.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logging.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

block_size = hyperparameters['block_size']
batch_size = hyperparameters['batch_size']
vocab_size = hyperparameters['vocab_size']
embedding_dim = hyperparameters['embedding_dim']
max_len = hyperparameters['max_len']
num_heads = hyperparameters['num_heads']
ffn_dim = hyperparameters['ffn_dim']
learning_rate = hyperparameters['learning_rate']
max_iters = hyperparameters['max_iters']
eval_interval = hyperparameters['eval_interval']
log_interval = hyperparameters['log_interval']
save_interval = hyperparameters.get('save_interval', 1000)  # Add save interval

model = MyModel()
model.to(device)
criterion = nn.CrossEntropyLoss()

# Initialize the optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Lists to store loss values for visualization
train_losses = []
val_losses = []

def save_model():
    torch.save(model.state_dict(), 'trained_model.pth')
    logging.info("Model saved to 'trained_model.pth'")

def emergency_stop(signal, frame):
    logging.info("Emergency stop triggered. Saving model...")
    save_model()
    sys.exit(0)

signal.signal(signal.SIGINT, emergency_stop)

def train(model, hyperparameters):
    model.train()
    try:
        for iter in range(hyperparameters['max_iters']):
            x, y = get_batch('train')
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate

            if iter % log_interval == 0:
                logging.info(f"Iteration {iter}, Loss: {loss.item()}")
                train_losses.append(loss.item())

            if iter % eval_interval == 0:
                val_loss = evaluate(model, hyperparameters)
                val_losses.append(val_loss)

            if iter % save_interval == 0:
                save_model()

    except KeyboardInterrupt:
        logging.info("Training interrupted. Saving model...")
        save_model()
        sys.exit(0)

    save_model()
    plot_losses()

def evaluate(model, hyperparameters):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        x, y = get_batch('val')
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        total_loss += loss.item()
    avg_loss = total_loss / len(x)
    logging.info(f"Validation Loss: {avg_loss}")
    model.train()
    return avg_loss

def plot_losses():
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_progress.png')
    plt.show()

if __name__ == "__main__":
    train(model, hyperparameters)