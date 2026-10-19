# Smart Parking Detection — Streamlit deployment

This folder is intended for a GitHub repository connected to Streamlit Community Cloud.

## Important distinction

- **Dataset ZIP**: used to train the model in Jupyter or Colab.
- **`best.pt`**: the trained model required by this Streamlit app to make predictions.
- **Parking image**: uploaded by an app user for inference.

The Streamlit app does not train from a dataset ZIP. Training a YOLO model on Streamlit Community Cloud is not recommended because its compute, memory, runtime, and storage are limited. Train locally in Jupyter or in Colab, export `best.pt`, then upload that file in the app's sidebar. `model_info.json` is optional.

## Deploy

1. Create a GitHub repository.
2. Upload `app.py`, `parking_core.py`, `requirements.txt`, `README.md`, and the `models/` folder if using a committed model.
3. On https://share.streamlit.io, create an app using that repository, branch `main`, and main file `app.py`.
4. Deploy. Open the app URL and upload your trained `best.pt` in the sidebar if it is not already in `models/best.pt`.
5. Upload a JPG/PNG parking-lot image and view/download the annotated result.

## Model upload choices

**Option A — upload in the app:** upload `best.pt` in the sidebar after deployment. This avoids committing model weights to GitHub, but the model upload is session-based and may need to be repeated in a new session/restart.

**Option B — include in repository:** place the files at `models/best.pt` and optionally `models/model_info.json`. This is convenient for small weights. GitHub rejects regular Git files larger than 100 MB; use Git LFS or external model storage for larger files. Do not upload secrets or private data.

## Local test

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The model file is not included in this package. Predictions require a trained Ultralytics YOLO `.pt` weights file compatible with the installed Ultralytics version.
