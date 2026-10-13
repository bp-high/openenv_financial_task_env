"""Media/content production: compose ABC notation meeting a given specification.

Derived from XiaomiMiMo/MiMo-V2.6-RL-oss `music` prompts (Apache-2.0): the model is asked for an ABC piece with a
stated key, meter, tempo, voice count and length, and a deterministic checker verifies the written notation. The
answer is the ABC text; grading is all-or-nothing on the stated constraints, so the model must emit valid,
constraint-matching notation rather than prose.
"""

import re

from .base import COMMON_RULES, Family, Instance

KEYS = ["C major", "G major", "D major", "A major", "F major", "Bb major", "Eb major",
        "A minor", "E minor", "D minor", "B minor", "G minor", "C minor"]
TEMPO_WORD = {"adagio": (55, 76), "andante": (76, 108), "moderato": (108, 120), "allegro": (120, 168)}
METERS = ["4/4", "3/4", "2/4", "6/8"]
ENSEMBLES = {1: ["solo piano", "solo cello", "solo guitar"],
             2: ["violin and cello duo", "flute and piano duo"],
             3: ["woodwind trio", "string trio"],
             4: ["string quartet", "brass quartet"]}
MOODS = ["a rainy harbour at dawn", "a neon rooftop at night", "a spring meadow", "a quiet library",
         "a long train journey", "a summer festival"]


def _beats_per_bar(meter: str) -> int:
    n, d = meter.split("/")
    return int(round(int(n) * (8 / int(d))))  # eighth-note units per bar (L:1/8)


class MusicAbc(Family):
    name = "media_music"
    title = "Compose a short piece in ABC notation to a brief"
    domain = "media and content production"
    levels = (1, 2)
    specs: dict = {}

    def generate(self, rng, level):
        key = rng.choice(KEYS)
        meter = rng.choice(METERS) if level >= 2 else "4/4"
        word = rng.choice(list(TEMPO_WORD))
        lo, hi = TEMPO_WORD[word]
        bpm = rng.randrange(lo, hi)
        voices = rng.choice([1, 2]) if level == 1 else rng.choice([1, 2, 3, 4])
        bars = rng.choice([8, 12, 16]) if level == 1 else rng.choice([16, 24, 32])
        ens = rng.choice(ENSEMBLES[voices])
        mood = rng.choice(MOODS)
        spec = {"key": key, "meter": meter, "bpm": bpm, "voices": voices, "bars": bars}
        tonic, mode = key.split()
        abc_key = tonic + ("m" if mode == "minor" else "")
        brief = (
            f"Compose a {ens} piece in {key}, {word} ({bpm} BPM), {meter} time, exactly {bars} bars, in "
            f"{voices} voice(s). Evoke {mood}. Return it as ABC notation in the answer with these exact header "
            f"lines: X:1, a T: title, M:{meter}, L:1/8, Q:1/4={bpm}, K:{abc_key}. After the headers, give exactly "
            f"{voices} voice(s); start each with its own line 'V:1' .. 'V:{voices}'. Each voice must contain "
            f"exactly {bars} bars separated by '|' (write the notes out; no repeat signs or ornament symbols). "
            'Submit {"answer": "<the full ABC text as one string>"}. ' + COMMON_RULES)
        inst = Instance(prompt=brief, files={"brief.txt": brief}, answer={"abc": self._oracle(spec)}, specs={})
        inst.grader = lambda a, _s=spec: self.check(_s, a.get("abc") if isinstance(a, dict) else a)
        inst._spec = spec
        return inst

    def _oracle(self, spec) -> str:
        tonic, mode = spec["key"].split()
        abc_key = tonic + ("m" if mode == "minor" else "")
        units = _beats_per_bar(spec["meter"])
        # one bar that fills exactly `units` eighth-notes using the tonic, chunked so each note length <= 8
        note = tonic[0].upper()
        rem, pieces = units, []
        while rem > 0:
            take = min(rem, 8)
            pieces.append(note + (str(take) if take > 1 else ""))
            rem -= take
        bar = "".join(pieces)
        music = " | ".join([bar] * spec["bars"]) + " |"
        lines = ["X:1", "T:Oracle", f"M:{spec['meter']}", "L:1/8", f"Q:1/4={spec['bpm']}", f"K:{abc_key}"]
        for v in range(1, spec["voices"] + 1):
            lines += [f"V:{v}", music]
        return "\n".join(lines) + "\n"

    def check(self, spec, abc) -> bool:
        if not isinstance(abc, str) or not (0 < len(abc) <= 20000):
            return False
        text = abc.replace("\r", "")

        def header(tag):
            m = re.search(rf"(?m)^{re.escape(tag)}:\s*(.+?)\s*$", text)
            return m.group(1).strip() if m else None

        if (header("M") or "").replace(" ", "") != spec["meter"]:
            return False
        if str(spec["bpm"]) not in (header("Q") or "").replace(" ", ""):
            return False
        tonic, mode = spec["key"].split()
        k = (header("K") or "").replace(" ", "")
        if not k.lower().startswith(tonic.lower()):
            return False
        is_minor = bool(re.search(r"m(in)?\b|min", k.lower()[len(tonic):]))
        if is_minor != (mode == "minor"):
            return False
        segments = re.split(r"(?m)^V:\s*\S+\s*$", text)[1:]
        if len(segments) != spec["voices"]:
            return False
        for seg in segments:
            music = "".join(l for l in seg.splitlines() if l and not re.match(r"^[A-Za-z]:", l))
            if len([b for b in music.split("|") if b.strip()]) != spec["bars"]:
                return False
        return True


FAMILIES = [MusicAbc()]
