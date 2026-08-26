import io
import re

import librosa
import soundfile as sf
import torch


WORD_COLORS = ["#E8A0A0", "#D45B5B", "#B02E2E", "#6E1414"]
TOKEN_COLORS = ["#9EC1E8", "#5B8FD4", "#2E5FB0", "#14356E"]


def _normalize_token_text(token_text):
    return token_text.replace("<|endoftext|>", "")
    

def _extract_token_records(processor, out, aligned_token_ids):
    token_records = []
    for step_logits, tok_id in zip(out.scores, aligned_token_ids):
        probabilities = torch.softmax(step_logits[0], dim=-1)
        token_records.append(
            {
                "token_text": _normalize_token_text(processor.decode([tok_id])),
                "prob": probabilities[tok_id].item(),
            }
        )
    return token_records


def _build_word_chunks(token_records, chunk_start_time, chunk_duration):
    words = []
    current = {"text": "", "probs": [], "token_texts": [], "start": None, "end": None}
    current_token_start_idx = None
    token_count = len(token_records) if token_records else 1
    token_duration = chunk_duration / token_count if token_count else 0.0

    for i, token in enumerate(token_records):
        token_text = token["token_text"]
        if not token_text and not current["text"]:
            continue

        if token_text.startswith(" ") and current["text"]:
            start_time = (
                chunk_start_time + (current_token_start_idx * token_duration)
                if current_token_start_idx is not None
                else chunk_start_time
            )
            end_time = chunk_start_time + (i * token_duration)
            current["start"] = round(start_time, 4)
            current["end"] = round(end_time, 4)
            words.append(current)
            current = {"text": "", "probs": [], "token_texts": [], "start": None, "end": None}
            current_token_start_idx = None

        if current_token_start_idx is None:
            current_token_start_idx = i

        current["text"] += token_text
        current["probs"].append(token["prob"])
        current["token_texts"].append(token_text)

    if current["text"]:
        start_time = (
            chunk_start_time + (current_token_start_idx * token_duration)
            if current_token_start_idx is not None
            else chunk_start_time
        )
        end_time = chunk_start_time + (token_count * token_duration)
        current["start"] = round(start_time, 4)
        current["end"] = round(end_time, 4)
        words.append(current)

    return words


def transcribe_with_token_data(audio_path, processor, model, device, language="de", chunk_length_s=28,
                                temperature=0.0):
    full_audio, sr = librosa.load(audio_path, sr=16000)
    chunk_samples = int(chunk_length_s * sr)
    words = []

    for chunk_start_sample in range(0, len(full_audio), chunk_samples):
        chunk_audio = full_audio[chunk_start_sample:chunk_start_sample + chunk_samples]
        if len(chunk_audio) < sr * 0.5:
            continue

        chunk_start_time = chunk_start_sample / sr
        _, speech_interval = librosa.effects.trim(chunk_audio, top_db=30)
        speech_start_offset = speech_interval[0] / sr
        speech_end_offset = speech_interval[1] / sr
        effective_chunk_start = chunk_start_time + speech_start_offset
        effective_duration = speech_end_offset - speech_start_offset
        inputs = processor(chunk_audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True).to(device)

        with torch.no_grad():
            out = model.generate(
                inputs["input_features"],
                language=language,
                task="transcribe",
                output_scores=True,
                return_dict_in_generate=True,
                attention_mask=inputs["attention_mask"],
                temperature=temperature,
                do_sample=(temperature > 0.0),
            )

        token_ids = out.sequences[0]
        offset = len(token_ids) - len(out.scores)
        aligned_token_ids = token_ids[offset:]
        token_records = _extract_token_records(processor, out, aligned_token_ids)

        words.extend(_build_word_chunks(token_records, effective_chunk_start, effective_duration))

    hypothesis = " ".join(word["text"].strip() for word in words if word["text"].strip())
    return hypothesis, words, full_audio, sr


def split_words_into_sentences(words):
    sentences = []
    current_sentence = []

    for word in words:
        current_sentence.append(word)
        if re.search(r"[.?!]\s*$", word["text"]):
            sentences.append(current_sentence)
            current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def compute_raw_word_stats(sent_words):
    stats = []
    for word in sent_words:
        probs = word["probs"]
        if not probs:
            continue

        stats.append(
            {
                "word": word["text"].strip(),
                "confidence_word": sum(probs) / len(probs),
                "confidence_token": min(probs),
                "token_texts": [t.strip() for t in word["token_texts"]],
                "token_probs": probs,
                "time_start": round(word["start"], 4) if word.get("start") is not None else None,
                "time_end": round(word["end"], 4) if word.get("end") is not None else None,
            }
        )
    return stats


def build_sentences(words):
    """
    Returns a list of sentence dictionaries, each self-contained:
    {sentence_id, text, confidence_avg, start, end, words: [...]} 
    """
    sentence_word_groups = split_words_into_sentences(words)
    sentences = []

    for sent_id, sent_words in enumerate(sentence_word_groups):
        word_stats = compute_raw_word_stats(sent_words)
        confidence_avg = (
            sum(word["confidence_word"] for word in word_stats) / len(word_stats)
            if word_stats else 0.0
        )

        sentences.append(
            {
                "sentence_id": sent_id + 1,
                "text": " ".join(word["word"] for word in word_stats),
                "confidence_avg": round(confidence_avg, 4),
                "start": word_stats[0]["time_start"] if word_stats else None,
                "end": word_stats[-1]["time_end"] if word_stats else None,
                "words": word_stats,
            }
        )

    return sentences


def get_threshold(mode, slider, relative_reference):
    """absolute: fixed cutoff. relative: percentage of the reference value."""
    if mode == "absolute":
        return slider / 100
    if mode == "relative":
        return relative_reference * (slider / 100)
    return slider / 100


def pick_color(value, threshold, palette):
    if value >= threshold:
        return None
    ratio = max(0.0, min(1.0, value / threshold if threshold > 0 else 0))
    if ratio >= 0.75:
        return palette[0]
    if ratio >= 0.5:
        return palette[1]
    if ratio >= 0.25:
        return palette[2]
    return palette[3]


def apply_flags(sentences, word_mode, word_pct, token_mode, token_pct):
    """
    Word threshold is relative to that word's OWN sentence average.
    Token threshold is relative to that word's own confidence.
    """
    for sentence in sentences:
        for word in sentence["words"]:
            word["word_threshold"] = get_threshold(word_mode, word_pct, sentence["confidence_avg"])
            word["word_flag"] = word["confidence_word"] < word["word_threshold"]
            word["token_threshold"] = get_threshold(token_mode, token_pct, word["confidence_word"])
            word["token_flag"] = word["confidence_token"] < word["token_threshold"]
    return sentences


def render_sentence_html(sentences, show_word=True, show_token=True):
    blocks = []
    for sentence in sentences:
        html_words = []
        for word in sentence["words"]:
            word_color = pick_color(word["confidence_word"], word["word_threshold"], WORD_COLORS) if show_word else None
            token_color = pick_color(word["confidence_token"], word["token_threshold"], TOKEN_COLORS) if show_token else None
            final_color = token_color or word_color

            title = f'word={word["confidence_word"]:.2f} token={word["confidence_token"]:.2f}'
            if final_color:
                html_words.append(f'<span style="color:{final_color};" title="{title}">{word["word"]}</span>')
            else:
                html_words.append(f'<span title="{title}">{word["word"]}</span>')

        sentence_label = (
            f'<span style="font-size: 0.8rem; color: #6b7280; margin-right: 0.5rem;">'
            f'Nr. {sentence["sentence_id"]} (conf= {sentence["confidence_avg"]:.2f}):</span>'
        )
        sentence_text = (
            f'<span style="font-style: italic; font-size: 1.2rem; line-height: 1.5;">'
            f'{" ".join(html_words)}</span>'
        )
        blocks.append(
            f'<div style="margin: 0 0 0.08rem 0; line-height: 1.25;">{sentence_label}{sentence_text}</div>'
        )
    return "".join(blocks)


def get_audio_clip_bytes(full_audio, sr, start, end):
    start_sample = int(start * sr) if start is not None else 0
    end_sample = int(end * sr) if end is not None else len(full_audio)
    clip = full_audio[start_sample:end_sample]

    buffer = io.BytesIO()
    sf.write(buffer, clip, sr, format="WAV")
    buffer.seek(0)
    return buffer


def build_export_rows(filename, sentences, word_mode, word_pct, token_mode, token_pct,
                       show_word=True, show_token=True):
    rows = []
    for sentence in sentences:
        for word in sentence["words"]:
            row = {
                "filename": filename,
                "sentence_id": sentence["sentence_id"],
                "word": word["word"],
                "sentence_confidence": sentence["confidence_avg"],
            }

            if show_word:
                row["word_confidence"] = round(word["confidence_word"], 4)
                row["word_mode"] = word_mode
                row["word_pct"] = word_pct
                row["word_threshold"] = round(word["word_threshold"], 4)
                row["word_flag"] = word["word_flag"]

            if show_token:
                row["token_mode"] = token_mode
                row["token_pct"] = token_pct
                row["token_threshold"] = round(word["token_threshold"], 4)
                row["token_flag"] = word["token_flag"]
                row["lowest_confidence_token"] = round(word["confidence_token"], 4)
                row["tokens"] = " | ".join(word["token_texts"])

            row["time_start"] = word["time_start"]
            row["time_end"] = word["time_end"]
            rows.append(row)

    return rows