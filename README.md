# Wheat expert system: FC + Certainty Factor evaluation

Reference implementation and simulation-based evaluation accompanying the manuscript
*A Web-Based Expert System for Pest and Disease Diagnosis in Tropical Wheat Using Forward Chaining and Certainty Factor: Formal Model and Evaluation*.

## What is here
- `wheat_es_experiments.py` – knowledge base (8 pests/diseases, 13 symptoms, 17 rules), the forward-chaining + CF engine, three baselines, and experiments E0–E3 (verification, knowledge-base ambiguity, Monte-Carlo consultations, sensitivity to MB/MD).
- `wheat_es_extensions.py` – baselines (naive Bayes, Dempster–Shafer), signed-CF negative evidence, and the information-gain adaptive interview (experiments E4–E6). Imports `wheat_es_experiments.py`.
- `results/` – `results.json`, `results_ext.json` and the figures used in the paper.

## Reproduce
```bash
pip install -r requirements.txt
python3 wheat_es_experiments.py   # base model, experiments E0-E3: results.json and fig_*.png
python3 wheat_es_extensions.py    # naive Bayes / Dempster-Shafer comparison, negative evidence, adaptive interview: results_ext.json, fig_adaptive.png
```
The random seed is fixed (`SEED` in the script), so the numbers in the paper are reproduced exactly.

## Scope and limits
Simulated consultations use the knowledge base itself as ground truth, so the results measure
internal consistency and robustness, not accuracy on real crops. The web application (PHP) is not part of this repository.
`results.json` also contains calibration and threshold analyses that are not reported in the paper. The extensions are evaluated in simulation only and are not implemented in the web prototype.

## Citation
Tamaji, M. S. Bhuwana, E. Dhaniswara, E. L. Talakua. (manuscript; add citation details when published).
