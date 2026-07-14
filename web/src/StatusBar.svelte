<script>
  // Ports ui/info/StatusBar.py: a transient status_label on the left,
  // a permanent name_label (active score/recording name) on the right.
  import { session } from "./sessionState.svelte.js";

  let name = $derived(session.noteData?.title ?? "untitled_recording");
</script>

<div class="status-bar">
  <span class="status-label">{session.statusMessage}</span>
  <span class="name-label">{name}</span>
</div>

<style>
  /* QStatusBar background is genuinely distinct (#2a2b2e), and its QLabel
     children aren't dimmed - full-strength text like everywhere else. */
  .status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 3px 10px;
    background: var(--bg-statusbar);
    font-size: 0.78rem;
    color: var(--text);
  }
</style>
