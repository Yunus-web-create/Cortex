"""Zero-cost media: narration and video built from free services on this machine.

Speech comes from Edge's neural voices (no key, no quota, no account) and video
is composed locally with ffmpeg from free generated stills plus that narration.
Nothing here bills anyone and nothing needs a card; the only cost is CPU time.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import httpx

EDGE_VOICES = {
    # OpenAI-ish names people type, mapped to real neural voices
    "alloy": "en-US-AriaNeural",
    "echo": "en-US-GuyNeural",
    "fable": "en-GB-SoniaNeural",
    "onyx": "en-US-AndrewNeural",
    "nova": "en-US-JennyNeural",
    "shimmer": "en-US-AvaNeural",
    "aria": "en-US-AriaNeural",
    "jenny": "en-US-JennyNeural",
    "guy": "en-US-GuyNeural",
    "sonia": "en-GB-SoniaNeural",
    "ryan": "en-GB-RyanNeural",
}
DEFAULT_VOICE = os.getenv("CORTEX_TTS_VOICE", "en-US-AriaNeural")
FFMPEG_TIMEOUT = int(os.getenv("CORTEX_FFMPEG_TIMEOUT", "240"))


def ffmpeg_exe() -> str | None:
    """System ffmpeg, else the static binary that ships with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def available() -> dict:
    try:
        import edge_tts  # noqa: F401
        tts = True
    except Exception:
        tts = False
    return {"speech": tts, "video": bool(ffmpeg_exe()) and tts}


def _voice(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return DEFAULT_VOICE
    if "Neural" in n:
        return n
    return EDGE_VOICES.get(n.lower(), DEFAULT_VOICE)


async def tts(text: str, *, voice: str = "", rate: str = "+0%") -> bytes:
    """Speech as MP3 bytes. Edge voices are free and need no credentials."""
    import edge_tts

    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        raise ValueError("nothing to say")
    communicate = edge_tts.Communicate(clean[:4000], _voice(voice), rate=rate)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            chunks.append(chunk["data"])
    data = b"".join(chunks)
    if len(data) < 512:
        raise RuntimeError("the voice service returned no audio")
    return data


_IMG_LOCK: asyncio.Lock | None = None
_IMG_AT = 0.0
IMG_GAP = float(os.getenv("CORTEX_IMG_GAP", "6"))     # the free endpoint allows ~1 per 6 s


def _img_lock() -> asyncio.Lock:
    global _IMG_LOCK
    if _IMG_LOCK is None:
        _IMG_LOCK = asyncio.Lock()
    return _IMG_LOCK


async def fetch_image(prompt: str, *, width: int = 1024, height: int = 1024,
                      seed: int | None = None, tries: int = 5) -> bytes:
    """One free still, politely: serialised, spaced out, retried on 402/429."""
    global _IMG_AT
    url = ("https://image.pollinations.ai/prompt/" + quote(prompt[:800]) +
           f"?width={int(width)}&height={int(height)}&nologo=true&model=flux")
    if seed is not None:
        url += f"&seed={int(seed)}"
    last = ""
    async with _img_lock():
        for attempt in range(tries):
            wait = IMG_GAP - (time.time() - _IMG_AT)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async with httpx.AsyncClient(timeout=180, follow_redirects=True) as c:
                    r = await c.get(url)
                _IMG_AT = time.time()
                if r.status_code == 200 and r.content:
                    return r.content
                last = f"HTTP {r.status_code}"
                if r.status_code not in (402, 429, 500, 502, 503, 504):
                    break
            except Exception as exc:
                _IMG_AT = time.time()
                last = str(exc)[:120]
            await asyncio.sleep(6 + attempt * 6)
    raise RuntimeError(f"image service unavailable ({last})")


async def _still(prompt: str, width: int, height: int, seed: int) -> bytes:
    return await fetch_image(prompt, width=width, height=height, seed=seed)


SHOTS = ("wide establishing shot", "medium shot, subject centred", "close-up detail",
         "low angle, dramatic light", "wide shot, golden hour")


def _motion(kind: str, frames: int) -> str:
    """Ken Burns move for one still: zoom in/out or a lateral pan."""
    d = max(2, frames)
    if kind == "in":
        return (f"zoompan=z='min(zoom+0.0012,1.12)':x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':d={d}:s={{w}}x{{h}}:fps=30")
    if kind == "out":
        return (f"zoompan=z='max(1.12-0.0012*on,1.0)':x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':d={d}:s={{w}}x{{h}}:fps=30")
    if kind == "right":
        return (f"zoompan=z='1.10':x='(iw-iw/zoom)*(on/{d})':y='ih/2-(ih/zoom/2)':"
                f"d={d}:s={{w}}x{{h}}:fps=30")
    return (f"zoompan=z='1.10':x='(iw-iw/zoom)*(1-on/{d})':y='ih/2-(ih/zoom/2)':"
            f"d={d}:s={{w}}x{{h}}:fps=30")


def _dims(aspect: str) -> tuple[int, int]:
    a = (aspect or "16:9").replace(" ", "")
    if a in ("9:16", "vertical", "portrait"):
        return 720, 1280
    if a in ("1:1", "square"):
        return 960, 960
    if a in ("4:3",):
        return 1024, 768
    return 1280, 720


async def _run(cmd: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=FFMPEG_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("ffmpeg timed out")
    if proc.returncode != 0:
        tail = (err or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg failed: " + " / ".join(tail))


async def _duration(exe: str, path: Path) -> float:
    """Read a media file's duration out of ffmpeg's banner."""
    import re as _re
    proc = await asyncio.create_subprocess_exec(
        exe, "-i", str(path), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    m = _re.search(rb"Duration: (\d+):(\d+):(\d+\.?\d*)", err or b"")
    if not m:
        return 0.0
    h, mi, sec = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(sec)


async def compose_video(prompt: str, *, seconds: int = 8, aspect: str = "16:9",
                        narration: str = "", voice: str = "") -> tuple[bytes, str]:
    """Still(s) -> camera moves -> concat -> voice-over. Returns (mp4, mode)."""
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg is not installed")

    audio = b""
    if narration.strip():
        audio = await tts(narration, voice=voice)

    seconds = max(4, min(int(seconds or 8), 30))
    w, h = _dims(aspect)
    base = (prompt or "a beautiful cinematic scene").strip()

    with tempfile.TemporaryDirectory(prefix="cortex_video_") as tmp:
        tmpdir = Path(tmp)

        if audio:
            vfile = tmpdir / "voice.mp3"
            vfile.write_bytes(audio)
            spoken = await _duration(exe, vfile)
            if spoken > seconds:
                seconds = min(30, int(spoken) + 1)

        # the free renderer allows roughly one image every few seconds, so a
        # short clip uses one still with several camera moves
        want = 1 if seconds <= 9 else 2 if seconds <= 20 else 3
        stills: list[Path] = []
        for i in range(want):
            seed = int(time.time()) % 100000 + i * 977
            text = f"{base}, {SHOTS[i % len(SHOTS)]}, cinematic, highly detailed"
            try:
                blob = await _still(text, int(w * 1.4), int(h * 1.4), seed)
            except Exception:
                if not stills:
                    raise
                break
            img = tmpdir / f"still{i}.png"
            img.write_bytes(blob)
            stills.append(img)

        shots = 4 if seconds <= 12 else 6
        per = seconds / shots
        frames = max(24, int(per * 30))
        moves = ("in", "right", "out", "left")
        clips = []
        for i in range(shots):
            img = stills[i % len(stills)]
            clip = tmpdir / f"clip{i}.mp4"
            motion = _motion(moves[i % len(moves)], frames).format(w=w, h=h)
            vf = (f"scale={int(w*1.4)}:-2,crop={int(w*1.4)}:{int(h*1.4)},"
                  f"{motion},format=yuv420p")
            await _run([exe, "-y", "-loglevel", "error", "-i", str(img), "-vf", vf,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                        "-t", f"{per:.2f}", str(clip)])
            clips.append(clip)

        listing = tmpdir / "clips.txt"
        listing.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
        silent = tmpdir / "silent.mp4"
        await _run([exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(silent)])

        final = tmpdir / "final.mp4"
        if audio:
            length = max(1.0, await _duration(exe, silent))
            await _run([exe, "-y", "-loglevel", "error", "-i", str(silent),
                        "-i", str(tmpdir / "voice.mp3"), "-filter_complex",
                        f"[1:a]apad=whole_dur={length:.2f}[a]",
                        "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "128k", "-t", f"{length:.2f}", str(final)])
        else:
            shutil.copy(silent, final)
        data = final.read_bytes()

    if len(data) < 10_000:
        raise RuntimeError("the composed video came out empty")
    return data, ("stills + voice-over" if audio else "stills")
