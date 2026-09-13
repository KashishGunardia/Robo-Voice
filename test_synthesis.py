import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'third_party/Matcha-TTS')

import torchaudio
import soundfile as sf
from cosyvoice.cli.cosyvoice import AutoModel

print("Loading model...")
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

#text = "Hello, this is a test of CosyVoice running locally on CPU."
text = "I wanted to check in and see how everything is going on your end. It's been a while since we last spoke, and I've been thinking about a few things I'd like to share with you."
prompt_text = "You are a helpful assistant.<|endofprompt|>what you are going through. I'm not even trying to match any of that right now but I think I've made it pretty clear and I don't know how many times he"

print("Generating speech...")
for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, 'asset/my_voice_clip.wav', stream=False)):
    sf.write(f'my_voice_output_{i}.wav', result['tts_speech'].squeeze(0).numpy(), cosyvoice.sample_rate)
    print(f"Saved my_voice_output_{i}.wav")

print("Done.")