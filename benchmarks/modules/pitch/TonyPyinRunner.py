from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from algorithms.Config import Config
from app_logic.NoteData import Note, NoteData


class TonyPyinRunner:
    """Run the Tony pYIN note backend through Sonic Annotator.

    Tony consumes the pYIN Vamp plugin's ``notes`` output. Calling that plugin
    with Sonic Annotator keeps this benchmark on the same backend instead of
    reimplementing pYIN's note HMM in this repository.
    """

    TRANSFORM_ID = "vamp:pyin:pyin:notes"
    PLUGIN_ID = "vamp:pyin:pyin"
    OUTPUT_ID = "notes"

    # Values from Tony's pYIN transform setup.
    STEP_SIZE = 256
    BLOCK_SIZE = 2048
    PRECISE_TIME = 0
    LOW_AMP_SUPPRESSION = 0.2
    ONSET_SENSITIVITY = 0.7
    PRUNE_THRESH = 0.1

    def __init__(self, config: Config) -> None:
        self.config = config

    def detect_notes(self, audio_path: str | Path) -> NoteData:
        binary = self._sonic_annotator_binary()
        self._verify_pyin_available(binary)
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found for Tony pYIN: {audio_path}")

        with tempfile.TemporaryDirectory(prefix="attune-tony-pyin-") as temp_dir:
            transform_path = Path(temp_dir) / "tony-pyin-notes.ttl"
            transform_path.write_text(self._transform_ttl(), encoding="utf-8")

            result = subprocess.run(
                [
                    str(binary),
                    "-q",
                    "-t",
                    str(transform_path),
                    "-w",
                    "csv",
                    "--csv-stdout",
                    "--csv-digits",
                    "10",
                    str(audio_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=self._subprocess_env(),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "sonic-annotator failed for Tony pYIN notes "
                    f"(exit {result.returncode}): {result.stderr.strip()}"
                )
            return self._parse_notes_csv(result.stdout)

    @classmethod
    def _sonic_annotator_binary(cls) -> Path:
        candidates = []
        for env_name in ("ATTUNE_SONIC_ANNOTATOR", "SONIC_ANNOTATOR"):
            value = os.environ.get(env_name)
            if value:
                candidates.append(Path(value).expanduser())

        found = shutil.which("sonic-annotator")
        if found:
            candidates.append(Path(found))

        candidates.extend(
            [
                Path("/Applications/Sonic Annotator.app/Contents/MacOS/sonic-annotator"),
                Path.home() / "Applications/Sonic Annotator.app/Contents/MacOS/sonic-annotator",
                Path.home() / "Desktop/sonic-annotator-1.7.0-macos/sonic-annotator",
                Path.home() / "Downloads/sonic-annotator-1.7.0-macos/sonic-annotator",
                Path("/private/tmp/sonic-annotator-1.7.0-macos/sonic-annotator"),
            ]
        )

        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate

        raise FileNotFoundError(
            "sonic-annotator was not found. Install it on PATH or set "
            "ATTUNE_SONIC_ANNOTATOR=/path/to/sonic-annotator."
        )

    @classmethod
    def _verify_pyin_available(cls, binary: Path) -> None:
        result = subprocess.run(
            [str(binary), "-l"],
            check=False,
            capture_output=True,
            text=True,
            env=cls._subprocess_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "could not list Vamp plugins with sonic-annotator "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )
        transforms = set(result.stdout.splitlines())
        if cls.TRANSFORM_ID not in transforms:
            raise RuntimeError(
                f"{cls.TRANSFORM_ID} is not available to sonic-annotator. "
                "Install the pYIN Vamp plugin pack."
            )

    @classmethod
    def _subprocess_env(cls) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("LANG", "en_US.UTF-8")
        env.setdefault("LC_ALL", "en_US.UTF-8")
        return env

    @classmethod
    def _transform_ttl(cls) -> str:
        return f"""@prefix xsd:      <http://www.w3.org/2001/XMLSchema#> .
@prefix vamp:     <http://purl.org/ontology/vamp/> .
@prefix :         <#> .

:transform a vamp:Transform ;
    vamp:plugin <http://vamp-plugins.org/rdf/plugins/pyin#pyin> ;
    vamp:step_size "{cls.STEP_SIZE}"^^xsd:int ;
    vamp:block_size "{cls.BLOCK_SIZE}"^^xsd:int ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "fixedlag" ] ;
        vamp:value "1"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "lowampsuppression" ] ;
        vamp:value "{cls.LOW_AMP_SUPPRESSION}"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "onsetsensitivity" ] ;
        vamp:value "{cls.ONSET_SENSITIVITY}"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "outputunvoiced" ] ;
        vamp:value "0"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "precisetime" ] ;
        vamp:value "{cls.PRECISE_TIME}"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "prunethresh" ] ;
        vamp:value "{cls.PRUNE_THRESH}"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "threshdistr" ] ;
        vamp:value "2"^^xsd:float ;
    ] ;
    vamp:output <http://vamp-plugins.org/rdf/plugins/pyin#pyin_output_notes> .
"""

    def _parse_notes_csv(self, text: str) -> NoteData:
        note_data = NoteData()
        note_index = 0
        for row in csv.reader(text.splitlines()):
            if len(row) < 4:
                continue
            try:
                start_time = float(row[1])
                duration = float(row[2])
                frequency = float(row[3])
            except ValueError:
                continue
            if not (
                frequency > 0
                and duration > 0
                and start_time == start_time
                and duration == duration
            ):
                continue

            end_time = start_time + duration
            note_data.write_note(
                Note(
                    i=note_index,
                    start_time=start_time,
                    end_time=end_time,
                    midi_num=[self.config.freq_to_midi(frequency)],
                )
            )
            note_index += 1
        return note_data
