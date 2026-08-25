# ASR Confidence Review Tool

The aim of this tool is to help the user spot words/tokens Whisper is less confident about in its transcription. To this end, two thresholds can be set & adjusted. 

**Confidence**: How certain the model is about each word and/or token. A token is a smaller piece of a word, produced by the model. 
  - Word-level confidence: How certain the model is about each word? 
  - Token-level confidence: Was there a moment inside a particular word, where the model was really unsure (even if it was confident about the word as a whole)? 

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
  - `lowest_confidence_token`: the confidence of the worst token in that word 
  - `tokens`: the word's individual sub-tokens, as split by the model 
  - `time_start` / `time_end`: approximate timing, useful for cutting out words/sentences eventually 

**Limitations**: sentence splitting is based on simple punctuation detection and may therefore not accommodate abbreviations or missing punctuation well


## Running it

```bash
git clone https://github.com/helene-sdl/Confidence_visualisation
cd Confidence_visualisation
pip install -r requirements.txt
streamlit run app.py
```

GPU is used automatically if available when running locally, otherwise falls back to CPU.


## Files

- `app.py`: the Streamlit UI: upload, settings, thresholds, display, export
- `confidence_logic.py`: the underlying logic (transcription, confidence computation,
  sentence splitting, coloring, CSV export), no Streamlit dependency 

## Live Demo
https://confidencevisualisation-h5sjpgtcvwkhvckcwrhq8r.streamlit.app/


Note: the deployed version uses `openai/whisper-small` rather than `large-v3`, due to
memory constraints on Streamlit Community Cloud's free tier, which is why transcription accuracy is expected to be notably lower. For full use, run locally with `large-v3` (see "Running it" above).
