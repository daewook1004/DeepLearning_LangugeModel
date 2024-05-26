import dataset
from model import CharRNN, CharLSTM
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0

    for input_seq, target_seq in trn_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        hidden = model.init_hidden(input_seq.size(0))
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)

        optimizer.zero_grad()
        output, hidden = model(input_seq, hidden)
        loss = criterion(output, target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    trn_loss = total_loss / len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            # Adjusting input_seq dimensions
            
            hidden = model.init_hidden(input_seq.size(0))
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)
            else:
                hidden = hidden.to(device)
            output, hidden = model(input_seq, hidden)
            loss = criterion(output, target_seq.view(-1))
            total_loss += loss.item()

    val_loss = total_loss / len(val_loader)
    return val_loss

def main():
    input_file = 'shakespeare_train.txt'
    sequence_length = 30
    batch_size = 64
    hidden_size = 128
    n_layers = 2
    dropout = 0.3

    num_epochs = 20
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Shakespeare(input_file)
    data_size = len(dataset)
    indices = list(range(data_size))
    split = int(data_size * 0.8)
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    size = len(dataset.char2idx)
    rnn_model = CharRNN(size, hidden_size, size, n_layers, dropout).to(device)
    lstm_model = CharLSTM(size, hidden_size, size, n_layers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_rnn_model_path = 'best_rnn_model.pth'
    best_lstm_model_path = 'best_lstm_model.pth'

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        rnn_trn_loss = train(rnn_model, trn_loader, device, criterion, rnn_optimizer)
        lstm_trn_loss = train(lstm_model, trn_loader, device, criterion, lstm_optimizer)

        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)

        train_losses.append((rnn_trn_loss, lstm_trn_loss))
        val_losses.append((rnn_val_loss, lstm_val_loss))

        print(f'Epoch {epoch+1}/{num_epochs}, RNN:  Training Loss: {rnn_trn_loss:.4f}, Validation Loss: {rnn_val_loss:.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, LSTM: Training Loss: {lstm_trn_loss:.4f}, Validation Loss: {lstm_val_loss:.4f}')

        # Check if the current RNN model is the best so far
        if rnn_val_loss < best_val_loss:
            best_val_loss = rnn_val_loss
            torch.save(rnn_model.state_dict(), best_rnn_model_path)
        
        # Check if the current LSTM model is the best so far
        if lstm_val_loss < best_val_loss:
            best_val_loss = lstm_val_loss
            torch.save(lstm_model.state_dict(), best_lstm_model_path)

    # Plotting losses
    rnn_train_losses, lstm_train_losses = zip(*train_losses)
    rnn_val_losses, lstm_val_losses = zip(*val_losses)
    

   
    # rnn training loss

    plt.figure(figsize=(12, 6))
    plt.plot(rnn_train_losses, label='RNN Training Loss')
   
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses for RNN')
    plt.savefig('loss_tran_plot_RNN.png')  # Save the plot as an image file
    plt.show()

    # rnn validation loss

    plt.figure(figsize=(12, 6))
    plt.plot(rnn_val_losses, label='RNN Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Losses for RNN')
    plt.savefig('loss_val_plot_RNN.png')  # Save the plot as an image file
    plt.show()





   # lstm training loss

    plt.figure(figsize=(12, 6))
    plt.plot(lstm_train_losses, label='LSTM Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses for LSTM')
    plt.savefig('loss_tran_plot_LSTM_plot.png')  # Save the plot as an image file
    plt.show()


    plt.figure(figsize=(12, 6))
    plt.plot(lstm_val_losses, label='LSTM Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Losses for LSTM')
    plt.savefig('loss_val_LSTM_plot.png')  # Save the plot as an image file
    plt.show()









if __name__ == '__main__':
    main()
