"""
voice_coach.py
══════════════
Pre-exercise voice instruction generation and caching.

The server can return text-only instructions without credentials. When
OPENAI_API_KEY is present and VOICE_PROVIDER=openai, it generates and caches
audio once per exercise/model/voice/script combination.
With VOICE_PROVIDER=auto, Windows Speech is used as an offline fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from exercises import Exercise, REGISTRY
from voice_cues import build_voice_script

log = logging.getLogger(__name__)

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"

_AUDIO_MIME = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


@dataclass(frozen=True)
class VoiceInstruction:
    exercise: str
    display_name: str
    language: str
    provider: str
    model: str
    voice: str
    audio_format: str
    text: str
    audio_url: Optional[str] = None
    audio_ready: bool = False
    cached: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def media_type_for_audio(path: Path) -> str:
    """Return a suitable media type for a cached audio file."""
    return _AUDIO_MIME.get(path.suffix.lower().lstrip("."), "application/octet-stream")


class VoiceCoach:
    """Build and optionally synthesize natural English setup instructions."""

    def __init__(self, cfg, audio_url_prefix: str = "/voice/audio") -> None:
        self.cfg = cfg
        self.audio_url_prefix = audio_url_prefix.rstrip("/")

    @property
    def cache_dir(self) -> Path:
        return Path(self.cfg.voice_cache_dir)

    def instruction(self, exercise: Exercise) -> VoiceInstruction:
        text = build_voice_script(exercise, language=self.cfg.voice_language)
        provider = self._resolve_provider()
        audio_format = "wav" if provider == "windows_sapi" else self.cfg.voice_tts_format
        base = self._base_instruction(exercise, text, provider, audio_format)

        if not self.cfg.voice_enabled:
            return self._with_reason(base, "voice coach is disabled")

        if provider == "none":
            return self._with_reason(base, "voice provider is set to none")

        audio_path = self._cache_path(exercise, text, provider, audio_format)
        audio_url = f"{self.audio_url_prefix}/{audio_path.name}"

        if audio_path.exists():
            return VoiceInstruction(
                **{
                    **base.to_dict(),
                    "audio_url": audio_url,
                    "audio_ready": True,
                    "cached": True,
                }
            )

        try:
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            if provider == "openai":
                api_key = os.getenv(OPENAI_API_KEY_ENV, "").strip()
                if not api_key:
                    return self._with_reason(
                        base,
                        "OPENAI_API_KEY is not set; returning text-only instructions",
                    )
                audio = self._generate_openai_audio(api_key=api_key, text=text)
                audio_path.write_bytes(audio)
            elif provider == "windows_sapi":
                self._generate_windows_sapi_audio(text=text, audio_path=audio_path)
            else:
                return self._with_reason(base, f"unsupported voice provider: {provider}")
        except Exception as exc:
            log.warning("Voice coach audio generation failed for %s: %s", exercise.value, exc)
            return self._with_reason(base, str(exc))

        return VoiceInstruction(
            **{
                **base.to_dict(),
                "audio_url": audio_url,
                "audio_ready": True,
                "cached": False,
            }
        )

    def cached_audio_path(self, filename: str) -> Optional[Path]:
        """Resolve a cached audio filename safely inside the configured cache dir."""
        if Path(filename).name != filename:
            return None

        root = self.cache_dir.resolve()
        candidate = (root / filename).resolve()
        if candidate.parent != root:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def _base_instruction(
        self,
        exercise: Exercise,
        text: str,
        provider: str,
        audio_format: str,
    ) -> VoiceInstruction:
        model = "windows-sapi" if provider == "windows_sapi" else self.cfg.voice_tts_model
        voice = "system-default" if provider == "windows_sapi" else self.cfg.voice_tts_voice
        return VoiceInstruction(
            exercise=exercise.value,
            display_name=REGISTRY[exercise].display_name,
            language=self.cfg.voice_language,
            provider=provider,
            model=model,
            voice=voice,
            audio_format=audio_format,
            text=text,
        )

    def _with_reason(self, instruction: VoiceInstruction, reason: str) -> VoiceInstruction:
        return VoiceInstruction(**{**instruction.to_dict(), "reason": reason})

    def _resolve_provider(self) -> str:
        provider = self.cfg.voice_provider
        if provider != "auto":
            return provider
        if os.getenv(OPENAI_API_KEY_ENV, "").strip():
            return "openai"
        return "windows_sapi"

    def _cache_path(self, exercise: Exercise, text: str, provider: str, audio_format: str) -> Path:
        payload = "|".join(
            [
                provider,
                self.cfg.voice_tts_model,
                self.cfg.voice_tts_voice,
                self.cfg.voice_language,
                audio_format,
                str(self.cfg.voice_tts_speed),
                self.cfg.voice_tts_style,
                exercise.value,
                text,
            ]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        filename = f"{exercise.value}_{digest}.{audio_format}"
        return self.cache_dir / filename

    def _generate_windows_sapi_audio(self, text: str, audio_path: Path) -> None:
        env = os.environ.copy()
        env["VOICE_TEXT"] = text
        env["VOICE_OUT"] = str(audio_path.resolve())
        env["VOICE_LANG"] = self.cfg.voice_language

        script = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.Volume = 100
try {
    $culture = [System.Globalization.CultureInfo]::GetCultureInfo($env:VOICE_LANG)
    $synth.SelectVoiceByHints(
        [System.Speech.Synthesis.VoiceGender]::NotSet,
        [System.Speech.Synthesis.VoiceAge]::Adult,
        0,
        $culture
    )
} catch {}
$synth.SetOutputToWaveFile($env:VOICE_OUT)
$synth.Speak($env:VOICE_TEXT)
$synth.Dispose()
"""
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown PowerShell error").strip()
            raise RuntimeError(f"Windows Speech failed: {detail[:240]}")

    def _generate_openai_audio(self, api_key: str, text: str) -> bytes:
        base_url = os.getenv(OPENAI_BASE_URL_ENV, "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/audio/speech"

        payload = {
            "model": self.cfg.voice_tts_model,
            "voice": self.cfg.voice_tts_voice,
            "input": text,
            "response_format": self.cfg.voice_tts_format,
            "speed": self.cfg.voice_tts_speed,
        }
        if self.cfg.voice_tts_model not in ("tts-1", "tts-1-hd") and self.cfg.voice_tts_style:
            payload["instructions"] = self.cfg.voice_tts_style

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": _AUDIO_MIME.get(self.cfg.voice_tts_format, "audio/*"),
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            detail = body[:240].replace("\n", " ")
            raise RuntimeError(f"OpenAI TTS HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI TTS request failed: {exc.reason}") from exc


def speak_instruction_async(exercise: Exercise, language: str = "en-US") -> None:
    """Speak an exercise setup script to the default Windows audio device."""
    text = build_voice_script(exercise, language=language)

    def _speak() -> None:
        env = os.environ.copy()
        env["VOICE_TEXT"] = text
        env["VOICE_LANG"] = language
        script = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.Volume = 100
try {
    $culture = [System.Globalization.CultureInfo]::GetCultureInfo($env:VOICE_LANG)
    $synth.SelectVoiceByHints(
        [System.Speech.Synthesis.VoiceGender]::NotSet,
        [System.Speech.Synthesis.VoiceAge]::Adult,
        0,
        $culture
    )
} catch {}
$synth.Speak($env:VOICE_TEXT)
$synth.Dispose()
"""
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=45,
                check=False,
            )
        except Exception:
            log.debug("Desktop voice instruction failed", exc_info=True)

    threading.Thread(target=_speak, daemon=True).start()
