# Spam Email Classifier — Beginner Project

Stack: Python, Flask, Pandas, scikit-learn, TF-IDF, Multinomial Naive Bayes, HTML/CSS, GitHub, Vercel.

## Run locally (Windows)
1. Install Python and VS Code.
2. Open this folder in VS Code.
3. In the terminal run:

   ```powershell
   python -m venv .venv
   .venv\\Scripts\\Activate.ps1
   python -m pip install -r requirements.txt
   python train_model.py
   python -m flask --app app run --debug
   ```

4. Open http://127.0.0.1:5000

If PowerShell blocks activation, use Command Prompt and run `.venv\\Scripts\\activate.bat`.

## Dataset
`spam.csv` must contain columns named `label` and `message`; labels must be `spam` or `ham`.
The included CSV is a tiny demonstration dataset so you can test the workflow. It is not large or diverse enough for reliable accuracy. Replace it with a larger, appropriately licensed dataset for a serious project and report precision, recall, F1-score, and a confusion matrix.

## Vercel
Train the model locally first, then include `spam_model.joblib` in the GitHub repository. Import the repository into Vercel and deploy. Runtime support, file-size limits, and deployment settings can change; consult:
- https://vercel.com/docs/functions/runtimes/python
- https://vercel.com/kb/guide/ship-a-flask-app-on-vercel
