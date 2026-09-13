import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'third_party/Matcha-TTS')

import io
import soundfile as sf
import whisper
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()

print("Loading Whisper...")
whisper_model = whisper.load_model("base")

print("Loading CosyVoice...")
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

REFERENCE_WAV = 'asset/my_voice_clip.wav'
REFERENCE_PROMPT = "You are a helpful assistant.<|endofprompt|>what you are going through. I'm not even trying to match any of that right now but I think I've made it pretty clear and I don't know how many times he"

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp_input.wav", "wb") as f:
        f.write(contents)
    result = whisper_model.transcribe("temp_input.wav")
    return {"text": result["text"]}

@app.post("/speak")
async def speak(text: str = Form(...)):
    for i, result in enumerate(cosyvoice.inference_zero_shot(text, REFERENCE_PROMPT, REFERENCE_WAV, stream=False)):
        sf.write("temp_output.wav", result['tts_speech'].squeeze(0).numpy(), cosyvoice.sample_rate)
        break
    return FileResponse("temp_output.wav", media_type="audio/wav")

@app.get("/")
async def health():
    return {"status": "running"}