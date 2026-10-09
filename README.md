# CMPSC 448 Midterm Project: LLM Identification

Name: Hang Shi
Date: October 11, 2026


# How to run the project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src/verify_raw.py
python src/preprocess.py
python src/vocab.py
python src/baseline.py
python src/run_sweep.py
python src/analyze.py
python src/stats.py

# Requirements
torch>=2.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65

# Limits to the project
There is 100 prompts per model, one temperature, three models.
RQ3 and RQ4 not implemented.