from algorithms.Config import Config


class NotePopupSV:
    """Popup payloads for the ScoreViewer's mistake annotations: title +
    labeled rows per mistake type, rendered by the viewer's JS layer when an
    annotated note / insertion marker is clicked. (The GuitarHero's Qt
    counterpart is NotePopupGH.)"""

    @classmethod
    def mistake_payload(cls, mistake) -> dict:
        mtype = mistake.type
        if mtype == "insertion":
            return {
                "category": "pitch",
                "type": mtype,
                "title": "Extra note!",
                "rows": [
                    {"label": "Target pitch", "value": "-"},
                    {"label": "Played pitch", "value": cls._note_pitch_names(mistake.user_note)},
                ],
            }
        if mtype == "deletion":
            return {
                "category": "pitch",
                "type": mtype,
                "title": "Missed note!",
                "rows": [
                    {"label": "Target pitch", "value": cls._note_pitch_names(mistake.midi_note)},
                    {"label": "Played pitch", "value": "-"},
                ],
            }
        if mtype == "substitution":
            target = cls._nearest_target_midi(mistake.user_note, mistake.midi_note)
            played = cls.primary_midi(mistake.user_note)
            title = "Too sharp!" if played is not None and target is not None and played > target else "Too flat!"
            return {
                "category": "pitch",
                "type": mtype,
                "title": title,
                "rows": [
                    {"label": "Target pitch", "value": cls._pitch_name(target)},
                    {"label": "Played pitch", "value": cls._pitch_name(played, cents=True)},
                ],
            }
        if mtype in ("early", "late"):
            return {
                "category": "timing",
                "type": mtype,
                "title": "Early!" if mtype == "early" else "Late!",
                "rows": [
                    {
                        "label": "Target onset",
                        "value": cls._seconds(mistake.midi_note.start_time),
                    },
                    {
                        "label": "Played onset",
                        "value": cls._seconds(mistake.user_note.start_time),
                    },
                ],
            }
        if mtype in ("long", "short"):
            return {
                "category": "timing",
                "type": mtype,
                "title": "Too long!" if mtype == "long" else "Too short!",
                "rows": [
                    {
                        "label": "Target duration",
                        "value": cls._seconds(cls._duration(mistake.midi_note)),
                    },
                    {
                        "label": "Played duration",
                        "value": cls._seconds(cls._duration(mistake.user_note)),
                    },
                ],
            }
        return {"category": "pitch", "type": mtype, "title": f"{mtype.title()}!", "rows": []}

    @staticmethod
    def primary_midi(note) -> float | None:
        """The note's first voiced pitch, or None (rests / empty notes)."""
        if note is None or not note.midi_num:
            return None
        midi = note.midi_num[0]
        return None if midi is None or midi < 0 else float(midi)

    @staticmethod
    def _duration(note) -> float:
        return max(0.0, float(note.end_time - note.start_time)) if note is not None else 0.0

    @staticmethod
    def _seconds(value: float) -> str:
        return f"{float(value):.2f}s"

    @classmethod
    def _nearest_target_midi(cls, user_note, score_note) -> float | None:
        """The chord member closest to what was played (chords list every
        simultaneous pitch, so compare against the nearest one)."""
        played = cls.primary_midi(user_note)
        if score_note is None or not score_note.midi_num:
            return None
        targets = [float(m) for m in score_note.midi_num if m is not None and m >= 0]
        if not targets:
            return None
        if played is None:
            return targets[0]
        return min(targets, key=lambda midi: abs(played - midi))

    @staticmethod
    def _pitch_name(midi: float | None, cents: bool = False) -> str:
        if midi is None or midi < 0:
            return "-"
        name = Config.get_note_name(midi)
        if not cents:
            return name
        cents_value = (float(midi) - round(float(midi))) * 100.0
        if abs(cents_value) < 0.5:
            return name
        return f"{name} {cents_value:+.0f}¢"

    @classmethod
    def _note_pitch_names(cls, note) -> str:
        if note is None or not note.midi_num:
            return "-"
        names = [
            cls._pitch_name(float(midi))
            for midi in note.midi_num
            if midi is not None and midi >= 0
        ]
        return "/".join(names) if names else "-"
