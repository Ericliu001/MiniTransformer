# Mini-GPT Training and Generation Guide

## Overview

Your codebase already supports training a model once and then generating text from the saved model without retraining!

## Workflow

### Step 1: Train the Model Once

```bash
python apps/train.py --data data/sample.txt --model_dir models/bible_model
```

**What this does:**
- Loads text from `data/sample.txt`
- Builds a vocabulary and tokenizer
- Trains the Mini-GPT model
- Saves the trained model to `models/bible_model/model.pkl`
- Saves the tokenizer to `models/bible_model/tokenizer.pkl`
- Saves training config to `models/bible_model/config.txt`

**Training Options:**
```bash
python apps/train.py \
  --data data/sample.txt \
  --model_dir models/my_model \
  --epochs 20 \
  --max_seq_length 128 \
  --d_model 128 \
  --num_heads 4 \
  --num_layers 4 \
  --tokenization word
```

### Step 2: Generate Text (No Retraining!)

Once trained, you can generate text as many times as you want without retraining:

**Option A: Single Prompt**
```bash
python apps/generate.py \
  --model_dir models/bible_model \
  --prompt "In the beginning" \
  --max_new_tokens 50 \
  --temperature 0.8
```

**Option B: Interactive Mode**
```bash
python apps/generate.py \
  --model_dir models/bible_model \
  --interactive
```

Then type prompts interactively:
```
Prompt: In the beginning
Generated: In the beginning God created the heaven and the earth...

Prompt: And God said
Generated: And God said, Let there be light...
```

## What Gets Saved

After training, your `models/bible_model/` directory contains:

1. **model.pkl** - Trained model weights (embeddings, parameters)
2. **tokenizer.pkl** - Vocabulary and tokenization mappings
3. **config.txt** - Training configuration for reference

These files allow the generation script to load the model without retraining.

## Files Involved

- **apps/train.py** - Training script (line 129: `model.save()`)
- **apps/generate.py** - Generation script (line 53: `MiniGPT.load()`)
- **apps/mini_gpt.py** - Model definition with save/load methods (lines 750-806)
- **apps/data_utils.py** - Tokenizer save/load utilities

## Key Code

### Saving (train.py:129-130)
```python
model.save(model_path)
save_tokenizer(tokenizer, tokenizer_path)
```

### Loading (generate.py:53-54)
```python
model = MiniGPT.load(model_path)
tokenizer = load_tokenizer(tokenizer_path)
```

## Example Full Workflow

```bash
# 1. Train once (takes a few minutes)
python apps/train.py --data data/sample.txt --model_dir models/bible_model --epochs 10

# 2. Generate many times (instant after loading)
python apps/generate.py --model_dir models/bible_model --prompt "In the beginning"
python apps/generate.py --model_dir models/bible_model --prompt "And God said"
python apps/generate.py --model_dir models/bible_model --prompt "Let there be"

# 3. Or use interactive mode
python apps/generate.py --model_dir models/bible_model --interactive
```

## Summary

Your system ALREADY supports:
- Training once and saving the model to disk
- Loading the saved model for generation without retraining
- Generating text multiple times from the same trained model

No code changes needed - the functionality is fully implemented!
