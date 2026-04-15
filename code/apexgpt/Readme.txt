This folder contains the manuscript-specific ApexGPT scripts.

Use the repository root README for the canonical release instructions:
- ../../README.md

In this release package:
- the Conda environment file is stored at ../../env/environment.yml
- the public ProGen2 base checkpoint should be downloaded into ./checkpoints_base/
- training configs write outputs to ./runs/
- packaged EMA checkpoints are exposed through ./output/*/best_mode_eval

Typical commands from this directory:
- python main_AMP_EMA.py config_AMP_EMA.json
- python main_AMP.py config_AMP.json
- python main_MDH_EMA.py config_MDH_EMA.json
- python main_MDH.py config_MDH.json
- python generated_AMP_EMA.py
- python generated_MDH_EMA.py
