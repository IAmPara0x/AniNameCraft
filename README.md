# AniNameCraft


AniNameCraft is a character name generator powered by RNN. It specializes in creating unique and diverse names for both male and female anime characters, so you never have to struggle with finding the perfect name for your characters again!

This project is inspired by [char-rnn](https://github.com/karpathy/char-rnn).


# Quickstart

```bash

$ git clone git@github.com:IAmPara0x/AniNameCraft.git
$ cd AniNameCraft
$ chmod +x ./main.py
$ ./main.py hi --gender Female greedy
```

# Usage

To generate new names, you'll need to provide a prefix and specify the gender for which you want to generate the name. The prefix can be of any length, from a single letter to a longer sequence of characters. For generating names for male characters, use the `--gender Male` flag, and for female characters, use the `--gender Female` flag.
You would also need to specify the sampling strategy, there are four sampling strategies available:

1. `greedy` select the most probabale next character
2. `topK --k 5` TopK sampling with default value of `k` being 5
3. `topP --p 0.9` TopP (nucleus sampling) with the default value of `p` being 0.9
4. `beam-search --beam-width 3 --beam-depth 6` Use beam search to sample new names



```bash


# Generate male character names starting with "ts" using topP

$ ./main.py ts --gender Male topP
# output: tsugayuki


# Generate female character names starting with "y" using beam search

$ ./main.py hi --gender Female beam-search --beam-width 5 --beam-depth 10
# output: ['hinon', 'hisaki', 'hirono', 'hirona', 'hiroka']

```

