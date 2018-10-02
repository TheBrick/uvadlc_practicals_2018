import argparse
import torch

from part3.train import finish_phrase
from part3.train import sample_model_randomly
from part3.dataset import TextDataset
from part3.model import TextGenerationModel

parser = argparse.ArgumentParser()
parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
parser.add_argument('--model', type=str, default="", help='Path to model file')
parser.add_argument('--phrase', type=str, default="", help='Phrase to finish')
parser.add_argument('--length', type=int, default=30, help='How many characters to generate')
parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature')
config = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # No parameter this time? Okay
dataset = TextDataset(config.txt_file, 10)

if not config.model:
    config.model = config.txt_file + ".model.pt"

model = torch.load(config.model)
model.lstm.flatten_parameters()

print(config.phrase + "...")

for i in range(5):
    if not config.phrase:
        text = sample_model_randomly(model, config.length, dataset, device, config.temperature)
    else:
        phrase = torch.tensor(dataset.convert_to_idxs(config.phrase), dtype=torch.long, device=device).view(1,-1)
        text = finish_phrase(model, phrase, config.length, dataset, device, config.temperature)
    print(text)


