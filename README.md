# Wireless Sensor Network Embedding using Reinforcement Learning

If you have [nix](https://nixos.org/nix/) installed (for dependency management), you can run the training with

```
nix-shell --run 'python3 experiments/default.py'
```

or specify any other experiment. Evaluation can be run with

```
nix-shell --run 'python3 evaluate.py <target dir> <pickled model>'
```
