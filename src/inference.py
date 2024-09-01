from .model import RNNModel, Tokenizer
import random
import torch

from typing import Literal



type Gender = Literal["Male"] | Literal["Female"]

class Inference:
    def __init__(self, model: RNNModel, tokenizer: Tokenizer, temperature: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def greedy_sampling(self, input_seed: str, gender: Gender) -> str:

        while True:
            x = torch.tensor(self.tokenizer.encode(input_seed, gender, max_len=len(input_seed))).reshape(1,-1).to(self.model.config.device)
            last_logits = self.model(x).squeeze()[-1]
            new_char_idx = self._get_prob_dist(last_logits).argmax().item()

            if new_char_idx == self.tokenizer.token2idx[self.tokenizer.special_tokens.end_token]:
                return input_seed

            input_seed = input_seed + self.tokenizer.idx2token[new_char_idx]


    def topK_sampling(self, input_seed: str, gender: Gender, k: int = 5) -> str:

        while True:
            x = torch.tensor(self.tokenizer.encode(input_seed, gender, max_len=len(input_seed))).reshape(1,-1).to(self.model.config.device)
            last_logits = self.model(x).squeeze()[-1]
            new_char_idx = random.choice(self._get_prob_dist(last_logits).topk(k).indices.tolist())

            if new_char_idx == self.tokenizer.token2idx[self.tokenizer.special_tokens.end_token]:
                return input_seed
            
            input_seed = input_seed + self.tokenizer.idx2token[new_char_idx]

    def nucleus_sampling(self, input_seed: str, gender: Gender, threshold_p: float = 0.85):

        while True:
            x = torch.tensor(self.tokenizer.encode(input_seed, gender, max_len=len(input_seed))).reshape(1,-1).to(self.model.config.device)
            last_logits = self.model(x).squeeze()[-1]
            candidates = self._get_prob_dist(last_logits).sort(descending=True)

            cum_p = 0
            next_char_candidates = []

            for (idx, p) in zip(candidates.indices,candidates.values):

                if threshold_p < cum_p:
                    break
                next_char_candidates.append(idx)
                cum_p += p

            new_char_idx = random.choice(next_char_candidates).item()
            if new_char_idx == self.tokenizer.token2idx[self.tokenizer.special_tokens.end_token]:
                return input_seed
            
            input_seed = input_seed + self.tokenizer.idx2token[new_char_idx]

    def beam_search(self, input_seed: str, gender: Gender, beam_width: int = 3) -> list[str]:


        beams = [input_seed]
        beam_scores = [0]

        outputs = []


        while len(beams) != 0:
            print(f"{len(beams)=}, {beams=}")

            xs = []
            for input_seed in beams:
                xs.append(self.tokenizer.encode(input_seed, gender, max_len=max(len(n) for n in beams)))

            xs = torch.tensor(xs).reshape(len(beams), -1).to(self.model.config.device)

            last_logits = self.model(xs)[:,-1]

            prob_dist = self._get_prob_dist(last_logits).sort(descending=True)
            candidates, candidates_score = prob_dist.indices, prob_dist.values



            ys = []
            for (beam_idx,score) in enumerate(beam_scores):

                beam = beams[beam_idx]
                beam_candidates = candidates[beam_idx]
                beam_candidates_score = candidates_score[beam_idx]

                for (candidate, candidate_score) in zip(beam_candidates, beam_candidates_score):
                    ys.append((beam + self.tokenizer.idx2token[candidate.item()], score + torch.log(candidate_score)))
            
            ys.sort(key = lambda c: c[1],reverse=True)
            
            ys = ys[:beam_width]
            
            for idx, (name, score) in enumerate(ys):

                end_token = self.tokenizer.special_tokens.end_token

                if name[-len(end_token):] == end_token:
                    outputs.append((name[:-len(end_token)],score.item()))
                    ys.pop(idx)

            beams = [y[0] for y in ys]

        return outputs

    def _get_prob_dist(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        t = probs**self.temperature
        return t / t.sum()
