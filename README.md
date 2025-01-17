# TIMEVIEW

This is the re-implementation of "Towards Transparent Time Series Forecasting" with my personal methods.

## Clone the Repository
Clone the repository using:
```bash
git clone https://github.com/krzysztof-kacprzyk/TIMEVIEW.git
```

## Installation
You can install all required dependencies using conda and the following command:
```bash
conda env create -n timeview --file environment.yml
```
This will also install `timeview` (the main module) in editable mode.

## ThirdDegree Model
To run experiments with the ThirdDegree model, use the following command:
```bash
python benchmark.py --datasets airfoil_log flchain_1000 stress-strain-lot-max-0.2 synthetic_tumor_wilkerson_1 --baselines ThirdDegree --n_trials 10 --n_tune 100 --seed 0 --device gpu
```

## GNN-based Encoder
To use the GNN-based encoder:
1. Replace the TTS' encoder with the GNN-based encoder by uncommenting the corresponding line in `timeview/model.py`.

## HyperTTS
To run experiments with the HyperTTS model, use the following command:
```bash
python benchmark.py --datasets beta_900_20 --baselines HyperTTS --n_trials 10 --n_tune 100 --seed 0 --device gpu --n_basis 5 --rnn_type lstm
```

## Citations
If you use this code, please cite using the following information:

*Kacprzyk, K., Liu, T. & van der Schaar, M. Towards Transparent Time Series Forecasting. The Twelfth International Conference on Learning Representations (2024).*

```bibtex
@inproceedings{Kacprzyk.TransparentTimeSeries.2024,
  title = {Towards Transparent Time Series Forecasting},
  booktitle = {The {{Twelfth International Conference}} on {{Learning Representations}}},
  author = {Kacprzyk, Krzysztof and Liu, Tennison and {van der Schaar}, Mihaela},
  year = {2024},
}
