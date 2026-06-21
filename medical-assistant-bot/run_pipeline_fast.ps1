# Fast medical assistant pipeline (smaller corpus + batch ST embeddings)

# Uses Ollama for chat only; indexing uses sentence-transformers on CPU.



$env:OLLAMA_MODELS = "F:\Ollama\models"

Set-Location $PSScriptRoot



Write-Host "Step 1: Fast ingest (5000 chatbot rows)..."

python scripts/ingest_documents.py --fast



Write-Host "Step 2: Fast vector index (sentence-transformers)..."

python scripts/build_index.py --fast



Write-Host "Step 3: Train intent classifier..."

python scripts/train_intent_classifier.py



Write-Host "Done. Start chatbot with: streamlit run app.py"


