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
  /* QTreeWidget has no background override of its own - it's the base
     QWidget color. Item text isn't dimmed by kind (RecordingTree.py sets no
     per-kind color/font); only indentation (14px) distinguishes a recording
     from its parent score. */
  .recording-tree {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    background: var(--bg-window);
    padding: 4px 0;
    font-size: 0.85rem;
  }
  .tree-item {
    display: block;
    width: 100%;
    background: none;
    border: none;
    font: inherit;
    color: var(--text);
    text-align: left;
    padding: 4px 10px;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .recording-item {
    padding-left: 24px;
  }
  /* QAbstractItemView::item:!selected:hover */
  .tree-item:hover:not(.selected) {
    background: var(--bg-item-hover);
  }
  /* QAbstractItemView::item:selected */
  .tree-item.selected {
    background: var(--bg-item-selected);
    color: var(--text);
  }
  .empty {
    color: var(--text-secondary);
    font-size: 0.8rem;
    padding: 8px 10px;
  }
</style>
