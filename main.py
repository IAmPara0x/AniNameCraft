#!/usr/bin/python3

from AniNameCraft.inference import Inference 
from AniNameCraft.model import RNNConfig,RNNModel, Tokenizer
from AniNameCraft.cli import parser

import pickle

import torch


if __name__ == "__main__":

    args = parser.parse_args()
    input_seed = args.input_name_seed
    gender = args.gender
    device = args.device
    
    with open("./weights/base_male_female_tokenizer.pkl", "rb") as f:
        tokenizer: Tokenizer = pickle.load(f)


    config = RNNConfig(vocab_size=len(tokenizer.token2idx), embed_dim=512, hidden_dim=1024, device=device)
    model = RNNModel(config).to(config.device)
    model.eval()
    model.load_state_dict(torch.load("./weights/base_male_female_RNN.pt", map_location=config.device, weights_only=True))

    with torch.no_grad():
        inference = Inference(model, tokenizer, temperature=1.0)

        if args.sampling_strategy == "greedy":
            name = inference.greedy_sampling(input_seed, gender)
            print(name)

        if args.sampling_strategy == "topK":
            name = inference.topK_sampling(input_seed, gender, k=args.k)
            print(name)

        if args.sampling_strategy == "topP":
            name = inference.nucleus_sampling(input_seed, gender, threshold_p=args.p)
            print(name)

        if args.sampling_strategy == "beam-search":
            name = inference.beam_search(input_seed, gender, beam_width=args.beam_width, beam_depth=args.beam_depth)
            print(name)
