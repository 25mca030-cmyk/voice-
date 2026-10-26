# VoiceOS Pipecat server

Separate Python service. It does **not** run inside Lovable — deploy it yourself.

## What it does
- Browser test calls on `wss://<host>/ws` and Vobiz phone streams on `wss://<host>/phone`.
- Pipeline: Silero VAD → STT (Deepgram / OpenAI / Sarvam) → LLM (Gemini / OpenAI / Claude) → ElevenLabs TTS, with interruptions.
- Claims each call from VoiceOS with its one-time token, runs agent tools through VoiceOS, and saves transcript, tokens, interruptions and latency back to VoiceOS.
- Every request to VoiceOS is signed: `X-Pipecat-Signature = hex(HMAC_SHA256(PIPECAT_SHARED_SECRET, "<unix-ts>.<body>"))`, 60 s window.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in
uvicorn main:app --port 8765
```

## Deploy (any Docker host with long-lived WebSockets + HTTPS)
Fly.io, Railway, Render, Pipecat Cloud or a VPS. Serverless functions with short timeouts will not work.
```bash
docker build -t voiceos-pipecat .
docker run -p 8765:8765 --env-file .env voiceos-pipecat
```
Sizing: ~1 vCPU / 1–2 GB RAM per 5–10 concurrent calls. Pick a region near callers (e.g. Mumbai).

## Connect to VoiceOS
In VoiceOS project secrets set:
- `PIPECAT_SERVER_URL` = `https://<your-host>`
- `PIPECAT_SHARED_SECRET` = same value as this server

Then pick a **Pipecat · …** model on an agent and press Test Agent.

## Sarvam voices
The voice catalogue includes 37 Bulbul v3 voices, 7 legacy v2 voices and 222 Bulbul v4 Flash personas. Saved short-name voices keep using v3; legacy voices use v2; personas use v4 Flash. V4 personas require their listed language (Hindi–English personas accept either Hindi or English).

Redeploy this folder to Render to activate the Sarvam phone-audio repair. Include `main.py`, `requirements.txt`, `sarvam_flash.py` and `sarvam-v4-voices.json`. The adapter uses Sarvam's HTTPS API for all three model versions, retains the saved voice and language, and supports Pipecat's speech context IDs. Existing call authentication, interruption handling and ElevenLabs remain unchanged. `/health` must include `revision: sarvam-phone-http-2026-10-24` after deployment.

Run the Python audio regression tests with `python -m unittest discover -s tests -v`.
