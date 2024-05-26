import torch.nn as nn
import torch 
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        # RNN layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers,  dropout=dropout, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Forward pass through RNN layer
        embedded = self.embedding(input)

        output, hidden = self.rnn(embedded, hidden)
    
        # Get output from the last time step
        output = self.fc(output.reshape(output.size(0) * output.size(1), output.size(2)))
        
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        initial_hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return initial_hidden

		

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Forward pass through LSTM layer
        embedded = self.embedding(input)
        
        # Get output from the last time step
        output, hidden = self.lstm(embedded, hidden)
    
        # Get output from the last time step
        output = self.fc(output.reshape(output.size(0) * output.size(1), output.size(2)))
        
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        initial_hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))
        return initial_hidden