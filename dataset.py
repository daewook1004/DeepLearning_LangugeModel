# import some packages you need here
import torch
from torch.utils.data import Dataset
import numpy as np

class Shakespeare(Dataset):
  
    def __init__(self, input_file):
        # Load Input file 
        with open(input_file, 'r') as f:
            self.text = f.read()
        # construt character dictionary 
        self.chars = sorted(list(set(self.text))) 
        self.char2idx = {ch: idx for idx, ch in enumerate(self.chars)} #문자에 고유한 인덱스 부여 
        self.idx2char = {idx: ch for idx, ch in enumerate(self.chars)} #인덱스에 해당하는 문자 매핑 
        
        self.data = [self.char2idx[ch] for ch in self.text]



    def __len__(self):

        return len(self.data) - 30

         
    def __getitem__(self, idx):

        input = torch.tensor(self.data[idx:idx+30], dtype=torch.long)
        target = torch.tensor(self.data[idx+1:idx+31],  dtype=torch.long)
    
        return input, target

        
        

if __name__ == '__main__':
    data = Shakespeare('shakespeare_train.txt')
    print("Number of unique characters:", len(data.chars))
    print("First 100 characters:", data.text[:100])
    print("First 100 indices:", data.data[:100])
    print("Character dictionary:", data.charToidx)
    print(f"Dataset length: {len(data)}")
    print(f"First sample input: {data[0][0]}")
    print(f"First sample target: {data[0][1]}")