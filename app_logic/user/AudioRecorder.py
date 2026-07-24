import queue
import threading

import numpy as np
import sounddevice as sd

from app_logic.user.ds.Recording import Recording

class AudioRecorder:
    _WRITER_STOP = object()

    def __init__(self, recording: Recording):
        # time variables
        self.t_0: float = 0
        self.t_curr: float = 0
        self._sample_cursor = 0
        self._adc_time_origin: float | None = None

        # important reference: to its parent user_data
        self.recording = recording
        self.sr = recording.audio_data.sr if recording else 44100 # default sr

        # The PortAudio callback only copies blocks into this queue. AudioData
        # writes, locks, growth, and deque extension happen on _writer_thread,
        # outside the real-time callback.
        self._chunk_queue = queue.SimpleQueue()
        self._writer_thread: threading.Thread | None = None
        self._writer_error: Exception | None = None
        self.input_overflow_count = 0
        # Callbacks only feed the writer once armed (run). A primed-but-unarmed
        # stream keeps the device warm during the count-in without buffering the
        # pre-record audio (see prime).
        self._armed = False

        self.stream = self._open_stream()

    def _open_stream(self):
        """Open an inactive input stream for the current sample rate."""
        return sd.InputStream(
            samplerate=self.sr,
            channels=1,
            callback=self._callback,
            # Let PortAudio use the device's robust callback latency. On the
            # built-in CoreAudio device the low/high difference is only ~9 ms,
            # while forcing low latency makes this Python callback miss input
            # deadlines and drop whole blocks.
            blocksize=0,
        )

    def load_recording(self, recording: Recording):
        """Load a new Recording object to record into."""
        self.stop()
        previous_sr = int(round(self.stream.samplerate))
        self.recording = recording
        self.sr = int(recording.audio_data.sr)
        self.t_0 = 0
        self.t_curr = 0
        if self.sr != previous_sr:
            self.stream.close()
            self.stream = self._open_stream()

    def prime(self):
        """Warm the input device ahead of capture. The first stream.start()
        after opening incurs a CoreAudio power-up + ADC-settle transient (~1 s
        of lag that then clears); called at the count-in so the stream is
        already streaming and settled by the record point. Callbacks are
        discarded until run() arms, so no pre-record audio is buffered."""
        self._armed = False
        if self.stream.stopped:
            try:
                self.stream.start()
            except Exception as exc:
                print(f"[AudioRecorder] prime failed: {exc}", flush=True)

    def run(self, start_time: float=0):
        """Arm capture at start_time. Reuses an already-primed (running, settled)
        stream so the device stays at steady state; only cold-starts the stream
        when it was never primed."""
        self._stop_writer()  # single writer; leave a primed stream running

        # keep track of the current start time
        self.t_0, self.t_curr = start_time, start_time
        self._sample_cursor = 0
        self._adc_time_origin = None
        self.input_overflow_count = 0
        self._writer_error = None
        self.recording.a2p_queue.init_start_time(start_time)

        # Fresh queue drops anything the primed stream captured before arming.
        self._chunk_queue = queue.SimpleQueue()
        self._writer_thread = threading.Thread(
            target=self._write_chunks,
            daemon=True,
        )
        self._writer_thread.start()
        self._armed = True  # callbacks now feed the writer

        # A primed count-in already left the stream running and settled; only a
        # cold start pays (and, on this record, exposes) the device warm-up.
        if self.stream.stopped:
            try:
                self.stream.start()
            except Exception:
                self._armed = False
                self._stop_writer()
                raise

    def stop(self):
        # A callback exception can make a stream inactive without marking it
        # stopped. Reset either state before a later start(). Also un-primes a
        # count-in that was cancelled before it armed.
        self._armed = False
        if not self.stream.stopped:
            self.stream.stop()
        self._stop_writer()

    def _stop_writer(self):
        thread = self._writer_thread
        if thread is None:
            return
        if thread.is_alive():
            # FIFO ordering makes the worker finish every copied callback block
            # before it sees the sentinel.
            self._chunk_queue.put(self._WRITER_STOP)
            thread.join()
        self._writer_thread = None
        if self._writer_error is not None:
            print(f"[AudioRecorder] writer failed: {self._writer_error}", flush=True)

    def _write_chunks(self):
        """Write copied callback blocks on a normal Python worker thread."""
        while True:
            item = self._chunk_queue.get()
            if item is self._WRITER_STOP:
                return
            indata, adc_time, overflowed, status_text = item
            try:
                if status_text:
                    print(status_text, flush=True)
                if overflowed:
                    self.input_overflow_count += 1
                self._write_chunk(indata, adc_time)
            except Exception as exc:
                self._writer_error = exc
                return

    def _write_chunk(self, indata: np.ndarray, adc_time: float | None):
        """Preserve the ADC sample timeline, including any dropped-input gap."""
        block_index = self._sample_cursor
        if adc_time is not None:
            if self._adc_time_origin is None:
                self._adc_time_origin = adc_time
            block_index = max(
                0,
                int(round((adc_time - self._adc_time_origin) * self.sr)),
            )
            # Ignore sub-sample timestamp noise.
            if abs(block_index - self._sample_cursor) <= 1:
                block_index = self._sample_cursor

        if block_index > self._sample_cursor:
            missing = block_index - self._sample_cursor
            silence = np.zeros(missing, dtype=np.float32)
            self.recording.write_data(
                silence,
                start_time=self.t_0 + self._sample_cursor / self.sr,
            )
            self._sample_cursor = block_index
        elif block_index < self._sample_cursor:
            overlap = self._sample_cursor - block_index
            if overlap >= len(indata):
                return
            indata = indata[overlap:]

        self.recording.write_data(
            indata,
            start_time=self.t_0 + self._sample_cursor / self.sr,
        )
        self._sample_cursor += len(indata)
        self.t_curr = self.t_0 + self._sample_cursor / self.sr

    def _callback(self, indata, frames, time, status):
        """
        Called by PortAudio for each captured block. Keep this real-time path
        limited to copying metadata/audio into the normal-priority writer queue.

        Args:
            indata: the audio data block that has been recorded
            frames: number of frames in the audio data block
            time: time information about the audio data block
            status: status information about the audio data block
        """
        try:
            if not self._armed:
                return  # primed (count-in): keep the device warm, discard audio
            block = np.array(indata[:, 0], dtype=np.float32, copy=True)
            adc_time = getattr(time, "inputBufferAdcTime", None)
            adc_time = float(adc_time) if adc_time is not None else None
            overflowed = bool(getattr(status, "input_overflow", False))
            status_text = str(status) if status else ""
            self._chunk_queue.put(
                (block, adc_time, overflowed, status_text)
            )
        except Exception as exc:
            # Never let a Python exception escape: sounddevice would terminate
            # all future callbacks and leave the stream inactive-but-not-stopped.
            self._writer_error = exc
