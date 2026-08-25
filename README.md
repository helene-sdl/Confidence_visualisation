# ASR Confidence Review Tool

The aim of this tool is to help the user spot words/tokens Whisper is less confident about in its transcription. To this end, two thresholds can be set & adjusted. 

**Model**: openai/whisper-large-v3 (via Hugging Face Transformers) 
**Confidence**: How certain the model is about each word and/or token. A token is a smaller piece of a word, produced by the model. 
  - Word-level cofnidence: How certain the model is about each word? 
  - Token-level confidence: Was there a moment inside a particular worde, where the model was really unsure (even if it was confident about the word as a whole)? 

**Color coding explanation**: Red for word-level confidence, blue for token-level confidence. 
    If a token's word is also flagged, blue takes precedence. 
    Four shades per color show how far below the threshold a value is, going in 25% increments, relative to the currently set threshold.
    Lightest = just below threshold, darkest = far below threshold.   

**Absolute vs. relative thresholds**:
  - *Absolute*: a fixed cutoff (e.g. 50% = 0.50 confidence), the same for every word/token in the file.
  - *Relative*: the word threshold is a percentage of that word's own sentence average; the token threshold is a percentage of that word's own confidence.

**CSV columns**:
  - `sentence_id`: which sentence (1-indexed) a word belongs to 
  - `word` / `sentence_confidence` / `word_confidence`: the text and confidence values 
  - `word_threshold` / `word_flag`: the word-level cutoff used and whether it was crossed 
  - `token_threshold` / `token_flag`: same, at the token level 
  - `lowest_confidence_token`: the confidence of the  worst token in that word 
  - `tokens`: the word's individual sub-tokens, as split by the model 
  - `time_start` / `time_end`: approximate timing, useful for cutting out words/sentences eventually 

**Limitations**: sentence splitting is based on simple punctuation detection and may therefore not accomodate abbreviations or missing punctuations well


## Running it

```bash
streamlit run app.py
```

Requires `torch`, `transformers`, `streamlit`, `librosa`, `soundfile`, and `pandas`.
GPU is used automatically if available, otherwise falls back to CPU.

## Files

- `app.py`: the Streamlit UI: upload, settings, thresholds, display, export
- `confidence_logic.py`: the underlying logic (transcription, confidence computation,
  sentence splitting, coloring, CSV export), no Streamlit dependency 
