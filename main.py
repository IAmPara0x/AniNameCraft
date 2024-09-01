import torch

from src.inference import Inference 
from src.model import RNNConfig,RNNModel, Tokenizer

import pickle


if __name__ == "__main__":
    
    with open("./weights/base_male_female_tokenizer.pkl", "rb") as f:
        tokenizer: Tokenizer = pickle.load(f)

    config = RNNConfig(vocab_size=len(tokenizer.token2idx), embed_dim=512, hidden_dim=1024, device="cuda:0")
    
    model = RNNModel(config).to(config.device)
    model.eval()
    model.load_state_dict(torch.load("./weights/base_male_female_RNN.pt", map_location=config.device, weights_only=True))

    with torch.no_grad():
        inference = Inference(model,tokenizer, temperature=1.0)
        print("Beam search")
        name = inference.beam_search("r", "Female", beam_width=3)
        print(name)
