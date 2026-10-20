# Wheat expert system: FC + Certainty Factor evaluation

Reference implementation and simulation-based evaluation accompanying the manuscript
*A Web-Based Expert System for Pest and Disease Diagnosis in Tropical Wheat Using Forward Chaining and Certainty Factor: Formal Model and Evaluation*.

## What is here
- `wheat_es_experiments.py` – knowledge base (8 pests/diseases, 13 symptoms, 17 rules), the forward-chaining + CF engine, three baselines, and experiments E0–E3 (verification, knowledge-base ambiguity, Monte-Carlo consultations, sensitivity to MB/MD).
- `results/` – `results.json` and the figures used in the paper.

## Reproduce
```bash
pip install -r requirements.txt
python3 wheat_es_experiments.py   # writes results.json and fig_*.png to the current directory
```
The random seed is fixed (`SEED` in the script), so the numbers in the paper are reproduced exactly.

## Scope and limits
Simulated consultations use the knowledge base itself as ground truth, so the results measure
internal consistency and robustness, not accuracy on real crops. The web application (PHP) is not part of this repository.
The script also writes additional analyses (calibration, threshold policy) to `results.json`; these are not reported in the paper.

## Citation
Tamaji, M. S. Bhuwana, E. Dhaniswara, E. L. Talakua. (manuscript; add citation details when published).
