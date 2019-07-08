let
  # pkgs = import <nixpkgs> {};
  pkgs = let
    nixpkgs-rev = "73392e79aa62e406683d6a732eb4f4101f4732be"; # nixos-unstable on 2019-07-03
  in import (builtins.fetchTarball {
    name = "nixpkgs-${nixpkgs-rev}";
    url = "https://github.com/nixos/nixpkgs/archive/${nixpkgs-rev}.tar.gz";
    # `git ls-remote https://github.com/nixos/nixpkgs-channels nixos-unstable`
  }) {};
  inherit (pkgs) lib;
in pkgs.mkShell {
  buildInputs = with pkgs; [
    (python3.withPackages(ps: with ps; [
      # use my forked version of baselines
      # (https://github.com/openai/baselines/pull/931 etc.)
      (baselines.overrideAttrs (attrs: {
        # src = lib.cleanSource /home/timo/repos/baselines; # for quick experimenting
        src = pkgs.fetchFromGitHub {
          owner = "timokau";
          repo = "baselines";
          sha256 = "0pr616ny8k4k1bh7y445js319bz6lwd78h3g4nc28p6anl1dry0v";
          rev = "18a2eadeec269f7f0289c5fbdc6be8be8ca7e2a2";
        };
      }))
      graph_nets
      matplotlib
      pydot
      ipython
      pycallgraph # profiling
      numpy
      scipy
    ]))
    graphviz # for pycallgraph profiling
  ];
  shellHook = ''
    export OPENAI_LOG_FORMAT=stdout,csv,tensorboard
    export OPENAI_LOGDIR=$PWD/logs/$(date -Is)
  '';
}
