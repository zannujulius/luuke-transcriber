import os
import tempfile
import whisperx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Whisper Diarization API")

DEVICE = "cpu"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type="int8")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
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
        result = model.transcribe(audio, batch_size=8)
        language = result["language"]

        # Step 2: align word timestamps
        align_model, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
        result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE)

        # Step 3: diarize (assign speakers)
        if HF_TOKEN:
            diarize_model = whisperx.DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = [
            {
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": seg.get("text", "").strip(),
            }
            for seg in result["segments"]
        ]

    finally:
        os.unlink(tmp_path)

    return JSONResponse({
        "language": language,
        "segments": segments,
        "text": " ".join(seg["text"] for seg in segments),
    })


@app.get("/health")
def health():
    return {"status": "ok"}
