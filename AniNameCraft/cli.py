import argparse

parser = argparse.ArgumentParser(description="A program with subcommands")
parser.add_argument("input_name_seed", help="prefix of any length for the name to begin with")
parser.add_argument("--gender", help="Custom greeting message", choices=["Male", "Female"])
parser.add_argument("--device", help="device on which to run model on", choices=["cpu", "cuda"], default="cpu")

subparsers = parser.add_subparsers(dest="sampling_strategy", description="choose a sampling strategy to generate the name.")

subparsers.add_parser("greedy", help="Greedy sampling strategy")

topK_parser = subparsers.add_parser("topK", help="topK sampling strategy")
topK_parser.add_argument("--k", type=int, help="top k candidates to randomly sample from", default=5)

topP_parser = subparsers.add_parser("topP", help="necleus sampling strategy")
topP_parser.add_argument("--p", type=float, help="sample from candidates with cumulative probability exceeding the given probability threshold", default=0.9)

beam_search_parser = subparsers.add_parser("beam-search", help="beam search sampling strategy")
beam_search_parser.add_argument("--beam-width", type=int, help="beam search width", default=3)
beam_search_parser.add_argument("--beam-depth", type=int, help="the maximum number of characters it should produce", default=6)
