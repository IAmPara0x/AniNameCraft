from dataclasses import dataclass
from .model import RNNModel, Tokenizer
import random
import torch

from typing import Literal


@dataclass
class Beam:

    input_seed: str
    score: float

    def add_token(self, new_token: str, token_score: float) -> "Beam":
        return Beam(self.input_seed + new_token, self.score + token_score)


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
            print(self._get_prob_dist(last_logits))
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

    def beam_search(self, input_seed: str, gender: Gender, beam_width: int = 3, beam_depth: int = 6) -> list[str]:


        beams: list[Beam] = [Beam(input_seed,0)]
        current_beam_depth = 0

        outputs: list[Beam] = []


        while len(beams) != 0:

            xs = torch.tensor( [self.tokenizer.encode(beam.input_seed, gender, max_len=len(input_seed) + current_beam_depth) for beam in beams]
                             ).reshape(len(beams), -1).to(self.model.config.device)

            last_logits = self.model(xs)[:,-1]
            prob_dist = self._get_prob_dist(last_logits)


            ys: list[Beam] = []

            for beam_idx, beam in enumerate(beams):

                beam_candidates = torch.log(prob_dist[beam_idx])
                for (candidate, candidate_score) in enumerate(beam_candidates):
                    ys.append(beam.add_token(self.tokenizer.idx2token[candidate], candidate_score.item()) )
       
            ys.sort(key=lambda c: c.score, reverse=True)
       
            # Get Top beams
            ys = ys[:beam_width]

            # Prune beams

            idx = 0
            while idx < len(ys):

                beam = ys[idx]
                end_token = self.tokenizer.special_tokens.end_token
        
                if beam.input_seed[-len(end_token):] == end_token:
                    outputs.append(Beam(beam.input_seed[:-len(end_token)],beam.score))
                    ys.pop(idx)
                elif beam_depth <= current_beam_depth:
                    outputs.append(beam)
                    ys.pop(idx)
                else:
                    idx += 1
       
            beams = ys
            current_beam_depth += 1
        
        outputs.sort(key=lambda b: b.score, reverse=True)
        return [b.input_seed for b in outputs[:beam_width]]

    def _get_prob_dist(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        t = probs**self.temperature
        return t / t.sum()
