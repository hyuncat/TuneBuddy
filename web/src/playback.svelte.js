// MIDI playback via js-synthesizer (fluidsynth compiled to WASM, run in an
// AudioWorklet) + simultaneous playback of the user's uploaded recording
// via a plain <audio> element - mirrors the desktop's Perform tab, which
// plays synthesized score MIDI (MidiPlayer) and, if the "User" checkbox is
// on, the recording's own audio (AudioPlayer) at the same time (see
// perform.py's start_playback). Practice tab's synth-only playback isn't
// relevant here - Practice (live pitch-matching) is out of scope for the
// upload-only web app.
//
// Architecture: builds a Standard MIDI File client-side from note_data
// (smf.js) and hands it to js-synthesizer's built-in SMF player rather than
// manually scheduling midiNoteOn/midiNoteOff - see smf.js's header comment
// for why. Channel mute/metronome toggles rebuild + reload the SMF (a small
// audible glitch on toggle, not on ordinary playback) rather than filtering
// events post-hoc, since the built-in player doesn't expose a per-message
// filter hook the way the desktop's hand-rolled scheduler does.
import { AudioWorkletNodeSynthesizer, Constants } from "js-synthesizer";
import { buildSMF, ticksToSeconds, secondsToTicks } from "./smf.js";
import { noteFromArray } from "./mistakes.js";

const SYNTH_BASE = "/synth";
const SOUNDFONT_URL = `${SYNTH_BASE}/MuseScore_General.sf3`;
const LIBFLUIDSYNTH_URL = `${SYNTH_BASE}/libfluidsynth-2.4.6-with-libsndfile.js`;
const WORKLET_URL = `${SYNTH_BASE}/js-synthesizer.worklet.js`;

const POLL_INTERVAL_MS = 100; // matches WallClock's own 10Hz cursor cadence

function createPlaybackState() {
  let ready = $state(false);
  let loading = $state(false);
  let error = $state("");
  let isPlaying = $state(false);
  let currentTime = $state(0);
  let duration = $state(0);

  // Toolbar's Playback menu state - "User" (recorded audio) defaults on,
  // every instrument channel defaults on, Metronome defaults on: all match
  // ui/info/Toolbar.py's checkbox/switch initial states.
  let userAudioOn = $state(true);
  let mutedChannels = $state(new Set());
  let metronomeOn = $state(true);

  let context = null;
  let synth = null;
  let sfontId = null;
  // Created eagerly (not inside ensureInit) since a plain <audio> element
  // doesn't need a user gesture to exist or load a src - only .play() does.
  // It used to be created lazily in ensureInit(), which only runs on the
  // first Play click; loadUserAudio() (called the moment a recording file
  // is picked, well before that) would see audioEl still null and silently
  // drop the file, leaving nothing for "User" playback to actually play.
  let audioEl = new Audio();
  audioEl.preload = "auto";
  let pollHandle = null;

  let currentNoteData = null;
  let currentBpm = 120;
  let currentMetronomeChannel = null;
  let lastLoadedKey = null; // dedupe rebuilds when nothing relevant changed

  function activeChannelsData() {
    if (!currentNoteData) return [];
    const channels = [];
    for (const chStr of Object.keys(currentNoteData.note_data ?? {})) {
      const channel = Number(chStr);
      if (channel === currentMetronomeChannel && !metronomeOn) continue;
      if (channel !== currentMetronomeChannel && mutedChannels.has(channel)) continue;
      const rawNotes = currentNoteData.note_data[chStr];
      if (!rawNotes || rawNotes.length === 0) continue;
      const notes = rawNotes.map(noteFromArray);
      channels.push({ channel, program: notes[0]?.instrument ?? 0, notes });
    }
    return channels;
  }

  // retrievePlayerTotalTicks() only returns a real value once playPlayer()
  // has actually been called at least once (verified empirically - it's 0
  // right after addSMFDataToPlayer resolves), which would leave the time
  // label showing 00:00.0 total until the user presses Play. Computing
  // duration directly from note_data instead makes it available the moment
  // a score loads, independent of the audio engine ever starting - and
  // matches every OTHER channel's notes too, not just the ones currently
  // audible, so muting a channel doesn't change the displayed length.
  function computeDuration(noteData) {
    let maxEnd = 0;
    for (const notes of Object.values(noteData?.note_data ?? {})) {
      for (const n of notes) {
        if (n[2] > maxEnd) maxEnd = n[2]; // [id, start, end, ...] - see JsonHandler._note_to_payload
      }
    }
    return maxEnd;
  }

  // Guards against a second concurrent call starting a duplicate WASM/
  // soundfont load if it's triggered twice before the first finishes (e.g.
  // loadNoteData's pre-warm firing again because the user swapped scores
  // quickly) - `synth` alone can't guard this since it's only set once the
  // whole async chain resolves, well after a second call could have
  // already started.
  let initPromise = null;
  function ensureInit() {
    if (synth) return Promise.resolve();
    if (initPromise) return initPromise;
    initPromise = (async () => {
      loading = true;
      error = "";
      try {
        context = new AudioContext();
        await context.audioWorklet.addModule(LIBFLUIDSYNTH_URL);
        await context.audioWorklet.addModule(WORKLET_URL);

        synth = new AudioWorkletNodeSynthesizer();
        synth.init(context.sampleRate);
        const node = synth.createAudioNode(context);
        node.connect(context.destination);

        const sfontBuffer = await (await fetch(SOUNDFONT_URL)).arrayBuffer();
        sfontId = await synth.loadSFont(sfontBuffer);

        ready = true;
      } catch (err) {
        error = err instanceof Error ? err.message : String(err);
      } finally {
        loading = false;
      }
    })();
    return initPromise;
  }

  // Rebuilds the SMF from noteData + current mute/metronome state and loads
  // it into the player, preserving playback position (and play/pause
  // state) across the reload. Called on first load and whenever mute state
  // changes; a no-op if nothing that affects the schedule actually changed.
  async function reload({ resume = false } = {}) {
    if (!synth || !currentNoteData) return;
    const key = JSON.stringify({
      title: currentNoteData.title,
      bpm: currentBpm,
      muted: [...mutedChannels].sort(),
      metronomeOn,
    });
    if (key === lastLoadedKey && !resume) return;
    lastLoadedKey = key;

    const wasPlaying = isPlaying;
    let resumeTicks = 0;
    if (wasPlaying || resume) {
      resumeTicks = secondsToTicks(currentTime, currentBpm);
      synth.stopPlayer();
      stopPolling();
    }

    await synth.resetPlayer();
    // pad to the full song length (computed across ALL channels, not just
    // audible ones) so muting a channel silences it without shortening
    // playback - see smf.js's buildSMF for why this matters.
    const smf = buildSMF(activeChannelsData(), currentBpm, duration);
    await synth.addSMFDataToPlayer(smf.buffer);

    if (resumeTicks > 0) {
      synth.seekPlayer(resumeTicks);
    }
    if (wasPlaying) {
      await synth.playPlayer();
      startPolling();
    }
  }

  // noteData: the /notedata (or /analyze-derived) payload. Called whenever
  // a new score is loaded - independent of Analyze, matching the desktop's
  // score-independent score loading.
  async function loadNoteData(noteData) {
    currentNoteData = noteData;
    currentBpm = noteData?.bpm ?? 120;
    currentMetronomeChannel = noteData?.metronome_channel ?? null;
    duration = computeDuration(noteData);
    lastLoadedKey = null;
    if (synth) {
      await reload();
    } else {
      // Fire off the WASM/soundfont load now instead of waiting for the
      // first Play click - none of this actually produces sound (that's
      // gated behind context.resume() in play(), which only runs from a
      // real click), so it doesn't need a user gesture and can run in the
      // background while the user is still picking a recording / reading
      // the score. By the time they hit Play, this has often already
      // finished, hiding the ~4s soundfont fetch+WASM-init cost entirely.
      ensureInit();
    }
  }

  function loadUserAudio(file) {
    if (!audioEl) return;
    if (audioEl.src) URL.revokeObjectURL(audioEl.src);
    audioEl.src = file ? URL.createObjectURL(file) : "";
  }

  function stopPolling() {
    if (pollHandle != null) {
      clearInterval(pollHandle);
      pollHandle = null;
    }
  }

  function startPolling() {
    stopPolling();
    pollHandle = setInterval(async () => {
      if (!synth) return;
      const ticks = await synth.retrievePlayerCurrentTick();
      // retrievePlayerCurrentTick can overshoot the track's real end by up
      // to a render-buffer's worth of ticks right as playback finishes
      // (fluidsynth processes audio in fixed-size chunks internally) -
      // clamp so the displayed time never reads past the known duration.
      currentTime = Math.min(ticksToSeconds(ticks, currentBpm), duration || Infinity);
      if (!synth.isPlayerPlaying()) {
        isPlaying = false;
        if (duration) currentTime = duration;
        stopPolling();
      }
    }, POLL_INTERVAL_MS);
  }

  async function play() {
    await ensureInit();
    if (!synth || !currentNoteData) return;
    if (lastLoadedKey == null) await reload();
    if (context.state === "suspended") await context.resume();

    // pressing Play right after reaching the end should restart, not
    // instantly re-finish from a position already at (or past) the end.
    if (duration && currentTime >= duration) {
      seek(0);
    }

    await synth.playPlayer();
    isPlaying = true;
    startPolling();

    if (userAudioOn && audioEl?.src) {
      audioEl.currentTime = currentTime;
      audioEl.play().catch(() => {}); // ignore autoplay/decoding races
    }
  }

  function pause() {
    if (!synth) return;
    synth.stopPlayer();
    isPlaying = false;
    stopPolling();
    audioEl?.pause();
  }

  function seek(seconds) {
    currentTime = seconds;
    if (synth && currentBpm) {
      synth.seekPlayer(secondsToTicks(seconds, currentBpm));
    }
    if (audioEl) audioEl.currentTime = seconds;
  }

  // Playback-speed only (see project notes): unlike the desktop's tempo
  // spinbox, this does NOT rescale note_data/ScoreData - it only changes
  // how fast the already-built SMF plays back, leaving analysis state
  // (mistake alignment, cursor time-base) untouched.
  function setTempo(bpm) {
    if (!synth) return;
    synth.setPlayerTempo(Constants.PlayerSetTempoType.ExternalBpm, bpm);
  }

  function setUserAudioOn(on) {
    userAudioOn = on;
    if (!on) audioEl?.pause();
    else if (isPlaying && audioEl?.src) {
      audioEl.currentTime = currentTime;
      audioEl.play().catch(() => {});
    }
  }

  function toggleChannelMute(channel) {
    const next = new Set(mutedChannels);
    if (next.has(channel)) next.delete(channel);
    else next.add(channel);
    mutedChannels = next;
    reload({ resume: true });
  }

  function setMetronomeOn(on) {
    metronomeOn = on;
    reload({ resume: true });
  }

  return {
    get ready() { return ready; },
    get loading() { return loading; },
    get error() { return error; },
    get isPlaying() { return isPlaying; },
    get currentTime() { return currentTime; },
    get duration() { return duration; },
    get userAudioOn() { return userAudioOn; },
    get mutedChannels() { return mutedChannels; },
    get metronomeOn() { return metronomeOn; },

    loadNoteData,
    loadUserAudio,
    play,
    pause,
    seek,
    setTempo,
    setUserAudioOn,
    toggleChannelMute,
    setMetronomeOn,
  };
}

export const playback = createPlaybackState();
