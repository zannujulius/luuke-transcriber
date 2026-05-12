import os
import tempfile
import whisperx
from whisperx.diarize import DiarizationPipeline
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

app = FastAPI(title="Whisper Diarization API")

DEVICE = "cpu"
HF_TOKEN = os.environ.get("HF_TOKEN", "REMOVED_TOKEN")
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
SPEAKER_LABELS = ["SPEAKER_1", "SPEAKER_2", "SPEAKER_3", "SPEAKER_4"]

print(f"WHISPER_MODEL: {MODEL_SIZE}")
print(f"HF_TOKEN set: {bool(HF_TOKEN)}")

model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type="int8")


def remap_speakers(segments: list) -> list:
    """Map SPEAKER_00, SPEAKER_01 ... to SPEAKER_1, SPEAKER_2 ..."""
    seen = {}
    for seg in segments:
        raw = seg.get("speaker")
        if raw and raw not in seen:
            idx = len(seen)
            seen[raw] = SPEAKER_LABELS[idx] if idx < len(SPEAKER_LABELS) else raw
    for seg in segments:
        raw = seg.get("speaker")
        if raw:
            seg["speaker"] = seen.get(raw, raw)
    return segments


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query(default=None, description="Force language code e.g. 'en'"),
    max_speakers: int = Query(default=2, ge=1, le=10, description="Max number of speakers"),
):
    allowed = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio = whisperx.load_audio(tmp_path)

        # Step 1: transcribe
        transcribe_kwargs = {"batch_size": 8}
        if language:
            transcribe_kwargs["language"] = language
        result = model.transcribe(audio, **transcribe_kwargs)
        detected_language = result["language"]

        # Step 2: align word timestamps
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_language, device=DEVICE
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, DEVICE,
            return_char_alignments=False
        )

        # Step 3: diarize and assign speakers
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set — diarization unavailable")

        diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = remap_speakers([
            {
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": seg.get("text", "").strip(),
            }
            for seg in result["segments"]
        ])

    finally:
        os.unlink(tmp_path)

    return JSONResponse({
        "language": detected_language,
        "num_speakers": len({s["speaker"] for s in segments if s["speaker"] != "UNKNOWN"}),
        "segments": segments,
        "text": " ".join(seg["text"] for seg in segments),
    })


@app.get("/health")
def health():
    return {"status": "ok"}
