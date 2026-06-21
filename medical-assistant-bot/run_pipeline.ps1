# Run full medical assistant training pipeline
# Uses Ollama models stored on F:\Ollama

$env:OLLAMA_MODELS = "F:\Ollama\models"
Set-Location $PSScriptRoot

Write-Host "Step 1: Ingest documents from Datafile..."
python scripts/ingest_documents.py

Write-Host "Step 2: Build vector index..."
python scripts/build_index.py

Write-Host "Step 3: Train intent classifier..."
python scripts/train_intent_classifier.py

Write-Host "Done. Start chatbot with: streamlit run app.py"
