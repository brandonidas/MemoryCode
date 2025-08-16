# Gemma 3 270M Evaluation - Sub-5K Context Branch

This branch is specifically configured for evaluating the MemoryCode dataset using the Gemma 3 270M model on dialogues with ≤5000 characters context.

## Setup

### 1. Dataset
- **Dataset Path**: `dataset_tiny/`
- **Dialogues**: 42 tiny context dialogues (≤5K characters each)
- **Total Cost**: ~$0.023 (99.5% reduction vs full dataset)

### 2. Model Configuration
- **Model**: `gemma-3-270m-it-qat-mlx`
- **API Endpoint**: `http://localhost:1234/v1/chat/completions`
- **Settings**: Temperature 0.7, no token limit

### 3. Quick Start

```bash
# Test API connection
python code/gemma_evaluation.py test_api_connection

# Run evaluation on tiny dataset
python code/gemma_evaluation.py evaluate_dataset

# Check results
python code/gemma_evaluation.py show_results
```

### 4. Example API Call
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-270m-it-qat-mlx",
    "messages": [
      { "role": "system", "content": "You are a software engineer. Generate only Python code." },
      { "role": "user", "content": "Write a function to calculate fibonacci numbers" }
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": false
  }'
```

### 5. Results Storage
- **Database**: `gemma_results.db`
- **Output Files**: `outputs/gemma/output_*.json`
- **Resume Support**: Yes, use `--resume` flag

### 6. Dataset Statistics
- 42 dialogues with ≤5000 characters each
- Average 1.3 sessions per dialogue
- 77 unique topics covered
- 4 different session types
- Estimated evaluation time: ~10-15 minutes

This lightweight setup enables rapid iteration and testing with the local Gemma model while maintaining representativeness of the full dataset.