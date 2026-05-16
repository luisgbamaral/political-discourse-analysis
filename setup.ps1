# Setup script for Political Discourse Analysis
# Creates a dedicated conda environment with Python 3.11 and installs all dependencies.
# Run from the project root: .\setup.ps1

$CONDA  = "C:\Users\luisg\anaconda3\Scripts\conda.exe"
$ENV_NAME = "political-discourse"

Write-Host "==> Creating conda environment '$ENV_NAME' with Python 3.11..."
& $CONDA create -n $ENV_NAME python=3.11 -y

# Resolve python.exe inside the new env
$PYTHON = "C:\Users\luisg\anaconda3\envs\$ENV_NAME\python.exe"
$PIP    = "C:\Users\luisg\anaconda3\envs\$ENV_NAME\Scripts\pip.exe"

Write-Host "==> Installing PyTorch (CPU build)..."
Write-Host "    If you have an NVIDIA GPU, replace the --index-url below with the CUDA wheel URL"
Write-Host "    from https://pytorch.org/get-started/locally/"
& $PIP install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

Write-Host "==> Installing remaining dependencies..."
& $PIP install `
    "anthropic>=0.40.0" `
    "python-dotenv==1.0.1" `
    "polars==1.2.1" `
    "pyarrow==16.1.0" `
    "pandas==2.2.2" `
    "transformers==4.41.2" `
    "tokenizers==0.19.1" `
    "sentencepiece==0.2.0" `
    "sacremoses==0.1.1" `
    "accelerate>=0.30.0" `
    "scikit-learn>=1.3.0" `
    "streamlit==1.36.0" `
    "plotly==5.22.0" `
    "tqdm==4.66.4" `
    "wordcloud==1.9.3" `
    "matplotlib==3.9.0"

Write-Host ""
Write-Host "==> Setup complete! Python: $PYTHON"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Copy .env.example to .env and fill in ANTHROPIC_API_KEY"
Write-Host "  2. In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter'"
Write-Host "     -> Enter path manually: $PYTHON"
Write-Host ""
Write-Host "Pipeline order:"
Write-Host "  & `"$PYTHON`" -m src.inference"
Write-Host "  & `"$PYTHON`" -m src.translator_gold"
Write-Host "  & `"$PYTHON`" -m src.labeler"
Write-Host "  & `"$PYTHON`" -m src.finetuner"
Write-Host "  & `"C:\Users\luisg\anaconda3\envs\$ENV_NAME\Scripts\streamlit.exe`" run app.py"
