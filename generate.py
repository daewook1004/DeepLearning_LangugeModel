import torch
import numpy as np
from model import CharRNN, CharLSTM
from dataset import Shakespeare

def generate_text(model, initial_input, length, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        hidden = model.init_hidden(1)
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)
        
        # Initialize input sequence
        input_sequence = [initial_input]
        input_idx = [model.char_to_idx[initial_input]]
        
        # Generate text
        for _ in range(length):
            input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(device)
            output, hidden = model(input_tensor, hidden)
            
            # Apply temperature and convert to probabilities
            output_dist = output.div(temperature).exp().cpu().squeeze().numpy()
            output_dist = output_dist / np.sum(output_dist)
            
            # Sample the next character
            sampled_idx = np.random.choice(len(output_dist), p=output_dist)
            input_idx = [sampled_idx]
            input_sequence.append(model.idx_to_char[sampled_idx])
        
        return ''.join(input_sequence)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the best model
    best_model_path = 'best_lstm_model.pth'
    dataset = Shakespeare('shakespeare_train.txt')
    
    # Initialize model
    n_chars = len(dataset.chars)
    hidden_size = 128
    n_layers = 2
    
    model = CharLSTM(n_chars, hidden_size, n_chars, n_layers).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.char_to_idx = dataset.char2idx
    model.idx_to_char = dataset.idx2char
    model.eval()
    
    # Set initial characters for text generation
    initial_inputs = ['F', 'K', 'N', 'S', 'U']

    temperatures = [0.3, 0.5, 0.7, 1.0, 1.5]
    
    # Generate text samples
    for temp in temperatures:
        print(f"Temperature: {temp}")
        for seed_char in initial_inputs:
            generated_text = generate_text(model, seed_char, length=100, temperature=temp)
            print(f"Seed Character: {seed_char} | Generated Text: {generated_text}\n")
            print("finish seed_char\n")

if __name__ == '__main__':
    main()