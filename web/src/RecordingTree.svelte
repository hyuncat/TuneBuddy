<script>
  // Simplified port of ui/info/RecordingTree.py's structure: a single-column
  // QTreeWidget (header hidden) showing the active score with its recording
  // as a child item. The desktop version also browses a folder library of
  // many scores/recordings - that's a persistence feature the web app
  // deliberately doesn't have (upload-only, no accounts/history), so this
  // only ever shows the one score/recording currently in the session rather
  // than pretending to browse a library that isn't there.
  import { session } from "./sessionState.svelte.js";

  let selected = $state("score"); // "score" | "recording" | null

  function selectScore() {
    selected = "score";
  }
  function selectRecording() {
    if (session.audioFile) selected = "recording";
  }
</script>

<div class="recording-tree">
  {#if session.noteData}
    <button
      type="button"
      class="tree-item score-item"
      class:selected={selected === "score"}
      onclick={selectScore}
    >
      {session.noteData.title}
    </button>
    {#if session.audioFile}
      <button
        type="button"
        class="tree-item recording-item"
        class:selected={selected === "recording"}
        onclick={selectRecording}
      >
        {session.audioFile.name}
      </button>
    {/if}
  {:else}
    <p class="empty">No score loaded. Use Upload above.</p>
  {/if}
</div>

<style>
  .recording-tree {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    background: var(--bg-surface);
    padding: 4px 0;
    font-size: 0.85rem;
  }
  .tree-item {
    display: block;
    width: 100%;
    background: none;
    border: none;
    font: inherit;
    color: inherit;
    text-align: left;
    padding: 4px 10px;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .recording-item {
    padding-left: 24px;
    color: var(--text-secondary);
  }
  .tree-item:hover {
    background: var(--bg-surface-raised);
  }
  .tree-item.selected {
    background: var(--accent);
    color: var(--bg-window);
  }
  .empty {
    color: var(--text-secondary);
    font-size: 0.8rem;
    padding: 8px 10px;
  }
</style>
