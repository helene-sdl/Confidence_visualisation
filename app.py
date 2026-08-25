import streamlit as st
import torch
import os
import tempfile
import pandas as pd
from datetime import datetime
import soundfile as sf
import io
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from confidence_logic import (
    transcribe_with_token_data,
    build_sentences,
    apply_flags,
    render_sentence_html,
    build_export_rows,
    get_audio_clip_bytes,
    WORD_COLORS,
    TOKEN_COLORS,
)

st.set_page_config(page_title="ASR Confidence Review Tool", layout="wide")

MAX_FILE_SIZE_MB = 200
MAX_DURATION_S = 600  # 10 minutes
MIN_SENTENCES_FOR_PLAYBACK = 2
MIN_SENTENCES_FOR_EXPANSION = 5

# Streamlit does not currently expose a padding argument for bordered
# containers. Keep this selector scoped to bordered-container wrappers so
# other layout blocks retain their default spacing.
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.5rem 0.75rem;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stMarkdownContainer"] p {
        margin: 0;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stMarkdownContainer"] {
        gap: 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return processor, model, device


def render_threshold_box(title, mode_key, pct_key, is_active, help_text):
    opacity = "1.0" if is_active else "0.4"
    st.markdown(f'<div style="opacity: {opacity};">', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f'<span style="opacity: {opacity};">{title}</span>', unsafe_allow_html=True)
        st.radio(
            "Mode", ["absolute", "relative"], key=mode_key, horizontal=True, index=1,
            help=help_text, disabled=not is_active,
        )
        st.slider("threshold %", 0, 100, 50, key=pct_key, disabled=not is_active)
    st.markdown('</div>', unsafe_allow_html=True)


st.title("ASR Confidence Review Tool")
st.caption("Upload audio, get word/token confidence per sentence, then export the results into a CSV.")

if "uploaded_files_data" not in st.session_state:
    st.session_state["uploaded_files_data"] = None

if "show_word" not in st.session_state:
    st.session_state["show_word"] = True
if "show_token" not in st.session_state:
    st.session_state["show_token"] = True
if "language_code" not in st.session_state:
    st.session_state["language_code"] = "de"
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.0


def get_loaded_files():
    return st.session_state.get("uploaded_files_data")


def get_selected_file(files):
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    selected_name = st.session_state.get("file_selector")
    if selected_name is None:
        selected_name = files[0].name

    for file in files:
        if file.name == selected_name:
            return file
    return files[0]


def reset_loaded_files():
    st.session_state["uploaded_files_data"] = None
    st.session_state.pop("file_selector", None)
    st.session_state.pop("add_files_uploader", None)


files = get_loaded_files()
uploaded_file = get_selected_file(files) if files else None

with st.expander(
    "Upload / manage files",
    expanded=get_loaded_files() is None,
):
    files = get_loaded_files()

    if files is None:
        st.markdown("**Choose audio file(s)**")
        uploaded = st.file_uploader(
            "Choose audio file(s)",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            st.session_state["uploaded_files_data"] = uploaded
            st.rerun()
    else:
        if len(files) == 1:
            st.markdown(f"**Current file:** {files[0].name}")
        else:
            st.markdown(f"**{len(files)} files loaded**")
            file_names = [f.name for f in files]
            st.selectbox("File to review", file_names, key="file_selector")

        more_files = st.file_uploader(
            "Add audio file(s)",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            key="add_files_uploader",
        )
        if more_files:
            existing_names = {f.name for f in files}
            new_files = [f for f in more_files if f.name not in existing_names]
            if new_files:
                st.session_state["uploaded_files_data"] = files + new_files
                st.rerun()

        if st.button("Replace loaded file(s)", type="primary"):
            reset_loaded_files()
            st.rerun()

uploaded_file = get_selected_file(get_loaded_files()) if get_loaded_files() else None

if uploaded_file is not None:
    with st.expander("Listen to audio"):
        st.audio(uploaded_file)
    with st.expander("Whisper Settings"):
        language_code = st.text_input(
            "Language code (e.g. de, en, fr)",
            value="de",
            key="language_code",
            help="ISO language code Whisper should assume for this audio (e.g. 'de' for German, 'en' for English). Does not translate. It only affects what language the transcription is written in.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("temperature", 0.0),
            step=0.1,
            key="temperature",
            help="Temperature controls randomness in the transcription. 0 = deterministic, higher values = more random. Use for experimentation.",
        )

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({file_size_mb:.1f} MB). Maximum supported size is {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    try:
        info = sf.info(io.BytesIO(uploaded_file.getvalue()))
        duration_seconds = info.frames / info.samplerate
        if duration_seconds > MAX_DURATION_S:
            st.error(f"Audio too long ({duration_seconds:.1f} s). Maximum supported length is {MAX_DURATION_S} seconds.")
            st.stop()
    except Exception:
        pass

    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_file.name)[1], delete=True) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp.flush()
            processor, model, device = load_model()
            with st.spinner("Transcribing..."):
                language_code = st.session_state.get("language_code", "de")
                temperature = st.session_state.get("temperature", 0.0)
                hypothesis, words, full_audio, sr = transcribe_with_token_data(
                    tmp.name,
                    processor,
                    model,
                    device,
                    language=language_code,
                    temperature=temperature,
                )
    except Exception as e:
        st.error("This file couldn't be transcribed. It may be corrupted or in an unsupported format.")
        st.exception(e)
        st.stop()

    if not words or not hypothesis.strip():
        st.warning("No speech detected in this file — the transcription came back empty.")
        st.stop()

    if len(hypothesis.strip()) < 3 and len(words) <= 1:
        st.warning("Very little content was transcribed — this file may contain music, silence, or no clear speech.")

    sentences = build_sentences(words)
    word_mode = st.session_state.get("word_mode", "relative")
    word_pct = st.session_state.get("word_pct", 50)
    token_mode = st.session_state.get("token_mode", "relative")
    token_pct = st.session_state.get("token_pct", 50)
    show_word = st.session_state.get("show_word", True)
    show_token = st.session_state.get("show_token", True)

    sentences = apply_flags(sentences, word_mode, word_pct, token_mode, token_pct)


    with st.expander("Threshold settings"):
        show_word = st.session_state.get("show_word", True)
        show_token = st.session_state.get("show_token", True)
        col_word_toggle, col_token_toggle = st.columns(2)


        with col_word_toggle:
            st.checkbox("Show word-level flags (🔴)", value=True, key="show_word")
        with col_token_toggle:
            st.checkbox("Show token-level flags (🔵)", value=True, key="show_token")
        col_word, col_token = st.columns(2)

        with col_word:
            render_threshold_box(
                '<b>Word-level threshold</b>', "word_mode", "word_pct",
                is_active=show_word,
                help_text="Absolute: fixed cutoff (e.g. 50% = 0.50 confidence). Relative: % of this word's own sentence average confidence.",
            )

        with col_token:
            render_threshold_box(
                '<b>Token-level threshold</b>', "token_mode", "token_pct",
                is_active=show_token,
                help_text="Absolute: fixed cutoff. Relative: % of that word's own confidence.",
            )

    with st.container(border=True):
            st.markdown("**Whisper Transcription:**")
            st.caption(
                f"{len(sentences)} sentence(s) detected · "
                f"word threshold {word_pct}% ({word_mode}) · token threshold {token_pct}% ({token_mode})"
            )

            can_play_sentence = len(sentences) >= MIN_SENTENCES_FOR_PLAYBACK
            can_expand_transcript = len(sentences) >= MIN_SENTENCES_FOR_EXPANSION
            if can_play_sentence and can_expand_transcript:
                playback_col, expand_col = st.columns(2)
            elif can_play_sentence or can_expand_transcript:
                playback_col = expand_col = st.container()
            else:
                playback_col = expand_col = None

            if can_play_sentence:
                with playback_col:
                    allow_sentence_playback = st.checkbox(
                        "Allow replay of individual sentences",
                        value=True,
                    )
            else:
                allow_sentence_playback = False

            if can_expand_transcript:
                with expand_col:
                    expand_transcript = st.checkbox("Expand transcript view", value=False)
            else:
                expand_transcript = False

            transcript_height = 900 if expand_transcript else 200
            with st.container(height=transcript_height):
                for sent in sentences:
                    sentence_col, play_col = st.columns([0.94, 0.06])

                    with sentence_col:
                        st.markdown(
                            render_sentence_html(
                                [sent],
                                show_word=show_word,
                                show_token=show_token,
                            ),
                            unsafe_allow_html=True,
                        )

                    with play_col:
                        if allow_sentence_playback:
                            st.button(
                                "▶",
                                key=f"play_sentence_{sent['sentence_id']}",
                                help=f"Play sentence {sent['sentence_id']}",
                                on_click=lambda sentence_id=sent["sentence_id"]:
                                    st.session_state.update(
                                        selected_sentence_id=sentence_id
                                    ),
                            )

            if allow_sentence_playback:
                selected_sentence_id = st.session_state.get("selected_sentence_id")
                selected_sentence = next(
                    (
                        sent for sent in sentences
                        if sent["sentence_id"] == selected_sentence_id
                    ),
                    None,
                )

                if selected_sentence is not None:
                    sentence_clip = get_audio_clip_bytes(
                        full_audio,
                        sr,
                        selected_sentence["start"],
                        selected_sentence["end"],
                    )


   
    
                    st.audio(sentence_clip, format="audio/wav")

    with st.expander("Word/Token scores"):
        rows_for_table = []
        for sent in sentences:
            for w in sent["words"]:
                row = {
                    "sentence_id": sent["sentence_id"],
                    "word": w["word"],
                    "sentence_confidence": sent["confidence_avg"],
                    "word_confidence": round(w["confidence_word"], 4),
                    "word_threshold": round(w["word_threshold"], 4),
                    "word_flag": w["word_flag"],
                    "token_threshold": round(w["token_threshold"], 4),
                    "token_flag": w["token_flag"],
                    "lowest_confidence_token": round(w["confidence_token"], 4),
                    "tokens": " | ".join(w["token_texts"]),
                    "time_start": w["time_start"],
                    "time_end": w["time_end"],
                }
                rows_for_table.append(row)
        st.dataframe(pd.DataFrame(rows_for_table), hide_index=True)

    export_rows = build_export_rows(
        uploaded_file.name, sentences, word_mode, word_pct, token_mode, token_pct,
        show_word=show_word, show_token=show_token,
    )
    export_df = pd.DataFrame(export_rows)
    st.download_button(
        "Download results for this file as CSV",
        export_df.to_csv(index=False, sep=";"),
        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_confidence_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

with st.expander("About this tool"):
    st.markdown("""
    The aim of this tool is to help the user spot words/tokens Whisper is less confident about in its transcription. To this end, two thresholds can be set & adjusted. 

    **Model**: openai/whisper-large-v3 (via Hugging Face Transformers) \n
    **Confidence**: How certain the model is about each word and/or token. A token is a smaller piece of a word, produced by the model. \n 
    - Word-level confidence: How certain the model is about each word? \n
    - Token-level confidence: Was there a moment inside a particular word, where the model was really unsure (even if it was confident about the word as a whole)? \n

    **Color coding explanation**: Red for word-level confidence, blue for token-level confidence. 
    If a token's word is also flagged, blue takes precedence. 
    Four shades per color show how far below the threshold a value is, going in 25% increments, relative to the currently set threshold.
    Lightest = just below threshold, darkest = far below threshold.   \n

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

    """)