import { App, Modal, Notice, Plugin, PluginSettingTab, Setting, requestUrl } from "obsidian";
import { StateEffect, StateField } from "@codemirror/state";
import { Decoration, DecorationSet, EditorView, ViewPlugin, ViewUpdate } from "@codemirror/view";

interface Suggestion {
  start: number;
  end: number;
  original: string;
  replacement: string;
  category: string;
  explanation_pt: string;
}
interface Analysis { original: string; suggestions: Suggestion[]; requires_approval: boolean }
interface CorrectionRecord {
  original: string; replacement: string; category: string;
  count: number; lastApplied: string;
}
interface TutorPreferences {
  ignoredWords: string[];
  dictionary: Record<string, string>;
  historyEnabled: boolean;
  history: CorrectionRecord[];
}
const norm = (value: string) => value.trim().toLocaleLowerCase();
function preferred(s: Suggestion, prefs: TutorPreferences): Suggestion | null {
  const key = norm(s.original);
  if (prefs.ignoredWords.includes(key)) return null;
  const value = prefs.dictionary[key];
  return value && value !== s.original ?
    { ...s, replacement: value, explanation_pt: "Tradução do seu dicionário pessoal." } :
    value === s.original ? null : s;
}


// Split long notes at whitespace boundaries, preserving absolute UTF-16 editor offsets.
// Bounded to avoid flooding a local 8 GB GPU with requests.
export function analysisChunks(text: string, maxLength = 500, maxChunks = 12): Array<{ start: number; text: string }> {
  const chunks: Array<{ start: number; text: string }> = [];
  let start = 0;
  while (start < text.length && chunks.length < maxChunks) {
    let end = Math.min(start + maxLength, text.length);
    if (end < text.length) {
      const span = text.slice(start, end);
      const breakAt = Math.max(span.lastIndexOf("\n"), span.lastIndexOf(" "));
      if (breakAt > maxLength / 2) end = start + breakAt + 1;
    }
    if (end <= start) break;
    chunks.push({ start, text: text.slice(start, end) });
    start = end;
  }
  return chunks;
}

const setSuggestions = StateEffect.define<Suggestion[]>();
const suggestionField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(decorations, tr) {
    if (tr.docChanged) {
      decorations = decorations.map(tr.changes).update({
        filter: (from, to, mark) => {
          const label = mark.spec.attributes?.title;
          const original = typeof label === "string" ? label.split(" → ")[0] : "";
          return original.length > 0 && tr.state.doc.sliceString(from, to) === original;
        }
      });
    }
    for (const effect of tr.effects) {
      if (effect.is(setSuggestions)) {
        decorations = Decoration.set(effect.value
          .filter(s => s.start >= 0 && s.end > s.start && s.end <= tr.state.doc.length)
          .sort((a, b) => a.start - b.start)
          .map(s => Decoration.mark({
            class: "pt-tutor-mark",
            attributes: { "data-category": s.category, title: s.original + " → " + s.replacement }
          }).range(s.start, s.end)), true);
      }
    }
    return decorations;
  },
  provide: field => EditorView.decorations.from(field)
});

class SuggestionModal extends Modal {
  constructor(app: App, private view: EditorView, private item: Suggestion,
              private snapshot: string,
              private ignore: (word: string) => Promise<void>,
              private saveWord: (word: string, replacement: string) => Promise<void>,
              private record: (item: Suggestion) => Promise<void>) { super(app); }
  onOpen() {
    const { contentEl, item } = this;
    contentEl.empty();
    contentEl.createEl("h3", { text: "PT English Tutor" });
    contentEl.createEl("p", { text: item.original + " → " + item.replacement });
    const explanationEl = contentEl.createEl("p", {
      text: item.explanation_pt || "Carregando explicação em português...",
      cls: "pt-tutor-modal-explanation"
    });
    if (!item.explanation_pt) {
      const line = this.view.state.doc.lineAt(item.start);
      const context = line.text.slice(0, 500);
      void requestUrl({
        url: "http://127.0.0.1:8765/v1/explain",
        method: "POST", contentType: "application/json",
        body: JSON.stringify({
          original: item.original, replacement: item.replacement,
          category: item.category, context
        }), throw: false
      }).then(response => {
        if (response.status !== 200) throw new Error("Explanation unavailable");
        const explanation = response.json?.explanation_pt;
        if (typeof explanation !== "string" || !explanation.trim()) throw new Error("Empty explanation");
        item.explanation_pt = explanation;
        if (explanationEl.isConnected) explanationEl.setText(explanation);
      }).catch(() => {
        if (explanationEl.isConnected) explanationEl.setText(
          "Não foi possível carregar a explicação. A correção continua disponível."
        );
      });
    }
    contentEl.createEl("button", { text: "Apply correction" }).addEventListener("click", () => {
      if (this.view.state.doc.toString() !== this.snapshot ||
          this.view.state.sliceDoc(item.start, item.end) !== item.original) {
        new Notice("Note changed; please wait for fresh suggestions.");
      } else {
        this.view.dispatch({ changes: { from: item.start, to: item.end, insert: item.replacement } });
        // Record only after the explicitly approved edit has been dispatched.
        void this.record(item);
      }
      this.close();
    });
    contentEl.createEl("button", { text: "Ignore once" }).addEventListener("click", () => this.close());
    contentEl.createEl("button", { text: "Always ignore" }).addEventListener("click", () => {
      void this.ignore(item.original); this.close();
    });
    contentEl.createEl("button", { text: "Add to dictionary" }).addEventListener("click", () => {
      void this.saveWord(item.original, item.replacement); this.close();
    });
  }
}

function createEditorExtension(app: App, prefs: TutorPreferences,
    ignore: (word: string) => Promise<void>,
    saveWord: (word: string, replacement: string) => Promise<void>,
    record: (item: Suggestion) => Promise<void>) {
  // Share approved model outputs across editor views within this plugin session.
  // A new note containing previously analyzed lines can display them immediately.
  const lineCache = new Map<string, Suggestion[]>();
  class TutorView {
    private timer: number | null = null;
    private sequence = 0;
    private suggestions: Suggestion[] = [];
    private snapshot = "";
    private lastSuccessfulQuery = "";
    private pending = new Set<string>();
    private lastFullScan = "";
    private fullScanRunning = false;
    private queuedFullScan = false;
    refreshPreferences() {
      this.suggestions = this.suggestions.map(s => preferred(s, prefs))
        .filter((s): s is Suggestion => s !== null);
      this.view.dispatch({ effects: setSuggestions.of(this.suggestions) });
    }
    scanCurrentNote(force = false) {
      const document = this.view.state.doc.toString();
      if (document.length < 30 || document.length > 12000) return;
      if (!force && document === this.lastFullScan) return;
      if (this.fullScanRunning) { this.queuedFullScan = true; return; }
      this.fullScanRunning = true;
      void this.analyzePastedText(document).then(completed => {
        if (completed) this.lastFullScan = document;
      }).catch(() => {
        new Notice("PT Tutor: document scan interrupted; reopen note to retry.");
      }).finally(() => {
        this.fullScanRunning = false;
        if (this.queuedFullScan) {
          this.queuedFullScan = false;
          this.scanCurrentNote(true);
        }
      });
    }

    constructor(private view: EditorView) {
      // Also inspect an existing note after its editor view becomes ready.
      const initial = view.state.doc.toString();
      if (initial.length >= 30 && initial.length <= 12000) {
        this.timer = window.setTimeout(() => this.scanCurrentNote(), 650);
      }
    }
    update(update: ViewUpdate) {
      if (!update.docChanged) return;
      this.sequence++;
      if (this.timer !== null) window.clearTimeout(this.timer);
      this.suggestions = this.suggestions.map(s => ({ ...s,
        start: update.changes.mapPos(s.start, 1),
        end: update.changes.mapPos(s.end, -1)
      })).filter(s => s.start < s.end &&
        update.state.doc.sliceString(s.start, s.end) === s.original);
      this.snapshot = update.state.doc.toString();
      let inserted = "";
      let cursor = update.state.selection.main.head;
      update.changes.iterChanges((_a, _b, _c, _d, value) => { inserted += value.toString(); });
      const last = inserted.slice(-1);
      if (!last) return;
      if (inserted.length >= 30) {
        // Pasted content is handled exclusively by the ordered full-note scan.
        this.scanCurrentNote(true);
        return;
      }
      if (this.fullScanRunning) return;
      // Analyze after punctuation, or after a short typing pause following any edit.
      // This also handles text pasted into the editor.
      const hasDelimiter = /[\s,.;:!?]/.test(last);
      const sentenceDone = /[.!?\n]/.test(last) || /[.!?]\s*$/.test(update.state.doc.sliceString(Math.max(0, cursor - 3), cursor));
      const current = this.sequence;
      // Completed words must be submitted immediately: a subsequent keystroke
      // must not cancel their request before it has even started.
      if (hasDelimiter) {
        void this.analyze(current, "word", cursor);
        if (sentenceDone) void this.analyze(current, "sentence", cursor);
      } else {
        this.timer = window.setTimeout(() => void this.analyze(current, "word", cursor), 900);
      }
    }
    private async analyzePastedText(document: string) {
      let offset = 0;
      let fenced = false;
      let yaml = document.startsWith("---\n");
      let processed = 0;
      let checked = 0;
      const jobs: Array<{ target: string; start: number }> = [];
      for (const line of document.split("\n")) {
        const lineStart = offset;
        offset += line.length + 1;
        if (yaml) { if (lineStart > 0 && line.trim() === "---") yaml = false; continue; }
        if (/^\s*(`{3,}|~{3,})/.test(line)) { fenced = !fenced; continue; }
        if (fenced || /^\s*(#{1,6}\s|>|[-*+]\s|\d+\.\s)/.test(line)) continue;
        if (line.includes("`") || line.includes("](")) continue;
        const target = line.trim();
        if (target.length < 5 || target.length > 500 || target.endsWith(":")) continue;
        if (processed >= 40) break;
        processed++;
        const start = lineStart + line.indexOf(target);
        jobs.push({ target, start });
      }
      // Analyze prose lines directly. A word-first pass delayed the first result.
      // Keep at most four concurrent requests and apply each result as it arrives.
      let index = 0;
      const worker = async () => {
        while (index < jobs.length) {
          const job = jobs[index++];
          if (await this.analyzePastedLine(job.target, job.start)) checked++;
        }
      };
      await Promise.all([worker(), worker(), worker(), worker()]);
      new Notice(`PT Tutor: checked ${checked}/${processed} text lines.`);
      return checked === processed;
    }
    private async analyzePastedLine(target: string, start: number) {
      try {
        const cacheKey = `batch:${target}`;
        let relative = lineCache.get(cacheKey);
        if (!relative) {
          const response = await requestUrl({
            url: "http://127.0.0.1:8765/v1/analyze", method: "POST",
            contentType: "application/json",
            body: JSON.stringify({ text: target, mode: "batch" }), throw: false
          });
          if (response.status !== 200) return false;
          const result = response.json as Analysis;
          if (!Array.isArray(result.suggestions)) return false;
          relative = result.suggestions;
          if (lineCache.size >= 200) lineCache.clear();
          lineCache.set(cacheKey, relative);
        }
        const latest = this.view.state.doc.toString();
        if (latest.slice(start, start + target.length) !== target) return false;
        const additions = relative
          .map(s => preferred(s, prefs)).filter((s): s is Suggestion => s !== null)
          .map(s => ({ ...s, start: s.start + start, end: s.end + start }))
          .filter(s => Number.isInteger(s.start) && Number.isInteger(s.end) &&
            s.start >= start && s.end <= start + target.length &&
            latest.slice(s.start, s.end) === s.original);
        const accepted = additions.filter(a => !this.suggestions.some(s =>
          s.category === "translation" && a.category !== "translation" &&
          a.start < s.end && a.end > s.start));
        this.suggestions = [...this.suggestions.filter(s =>
          !accepted.some(a => a.start < s.end && a.end > s.start)), ...accepted];
        this.snapshot = latest;
        this.view.dispatch({ effects: setSuggestions.of(this.suggestions) });
        return true;
      } catch { return false; }
    }
    private async analyze(sequence: number, mode: "word" | "sentence", cursor: number) {
      if (this.fullScanRunning) return;
      const document = this.view.state.doc.toString();
      // Skip protected Markdown only at the cursor, not the entire note.
      // A code fence elsewhere must not disable tutoring in long notes.
      const before = document.slice(0, cursor);
      if (document.startsWith("---\n")) {
        const frontmatterEnd = document.indexOf("\n---", 4);
        if (frontmatterEnd < 0 || cursor <= frontmatterEnd + 4) return;
      }
      const lines = before.split("\n");
      const fenceCount = lines.filter(line => /^\s*(`{3,}|~{3,})/.test(line)).length;
      if (fenceCount % 2 === 1) return;
      const currentLine = lines[lines.length - 1];
      if ((currentLine.match(/(?<!`)`(?!`)/g) ?? []).length % 2 === 1) return;
      const boundary = mode === "word" ? /[\s,.;:!?]/g : /[.!?\n]/g;
      const trimmed = before.replace(/[\s,.;:!?]+$/, "");
      let start = 0;
      for (const match of trimmed.matchAll(boundary)) start = match.index + match[0].length;
      const target = trimmed.slice(start);
      if (!target.trim() || target.length > 500 || target.length < 2) return;
      // Never send a sentence that crosses Markdown code or frontmatter boundaries.
      if (/```|~~~|^---$/m.test(target)) return;
      const queryKey = `${mode}:${start}:${target}`;
      if (queryKey === this.lastSuccessfulQuery || this.pending.has(queryKey)) return;
      this.pending.add(queryKey);
      try {
        const response = await requestUrl({
          url: "http://127.0.0.1:8765/v1/analyze",
          method: "POST",
          contentType: "application/json",
          body: JSON.stringify({ text: target, mode }),
          throw: false
        });
        // Later typing must not discard a completed word's response.
        // Accept only if the analyzed text is still unchanged at its offsets.
        if (this.fullScanRunning) return;
        const latestDocument = this.view.state.doc.toString();
        // Reject a stale response if anything before its original span changed.
        // Changes after the word are safe and should not cancel its underline.
        if (latestDocument.slice(0, start + target.length) !== document.slice(0, start + target.length)) return;
        if (mode === "word" && /[A-Za-z\u00c0-\u024f]/.test(latestDocument.charAt(start + target.length))) return;
        if (response.status !== 200) return;
        const result = response.json as Analysis;
        if (!Array.isArray(result.suggestions)) return;
        this.lastSuccessfulQuery = queryKey;
        const additions = result.suggestions
          .map(s => preferred(s, prefs)).filter((s): s is Suggestion => s !== null)
          .map(s => ({ ...s, start: s.start + start, end: s.end + start }))
          .filter(s => Number.isInteger(s.start) && Number.isInteger(s.end) &&
            s.start >= start && s.end <= start + target.length &&
            latestDocument.slice(s.start, s.end) === s.original);
        // A later sentence pass must never overwrite an already verified translation.
        const safeAdditions = additions.filter(a => !this.suggestions.some(s =>
          s.category === "translation" && a.category !== "translation" &&
          a.start < s.end && a.end > s.start));
        this.suggestions = [...this.suggestions.filter(s =>
          !safeAdditions.some(a => a.start < s.end && a.end > s.start &&
            !(a.category !== "translation" && s.category === "translation"))), ...safeAdditions];
        this.snapshot = latestDocument;
        this.view.dispatch({ effects: setSuggestions.of(this.suggestions) });
      } catch {
        // Keep editing responsive when the local AI is temporarily unavailable.
      } finally {
        this.pending.delete(queryKey);
      }
    }
    findAt(position: number) {
      return this.suggestions.find(s => position >= s.start && position < s.end);
    }
    getSnapshot() { return this.snapshot; }
    destroy() {
      this.sequence++;
      if (this.timer !== null) window.clearTimeout(this.timer);
    }
  }
  const extension = ViewPlugin.fromClass(TutorView, {
    eventHandlers: {
      focus(_event, view) {
        // Obsidian can reuse an editor view when reopening a note.
        view.plugin(extension)?.scanCurrentNote();
        return false;
      },
      click(event, view) {
        const target = event.target;
        if (!(target instanceof HTMLElement) || !target.closest(".pt-tutor-mark")) return false;
        const position = view.posAtCoords({ x: event.clientX, y: event.clientY });
        if (position === null) return false;
        const instance = view.plugin(extension);
        const suggestion = instance?.findAt(position);
        if (!instance || !suggestion) return false;
        new SuggestionModal(app, view, suggestion, instance.getSnapshot(), ignore, saveWord, record).open();
        return true;
      }
    }
  });
  return extension;
}

export default class PTEnglishTutor extends Plugin {
  preferences: TutorPreferences = { ignoredWords: [], dictionary: {}, historyEnabled: false, history: [] };
  private tutorExtension!: ReturnType<typeof createEditorExtension>;
  refreshPreferencesViews() {
    this.app.workspace.iterateAllLeaves(leaf => {
      const cm = (leaf.view as unknown as { editor?: { cm?: EditorView } }).editor?.cm;
      cm?.plugin(this.tutorExtension)?.refreshPreferences();
    });
  }
  async addIgnored(word: string) {
    const key = norm(word);
    if (!key || key.length > 100) return;
    if (!this.preferences.ignoredWords.includes(key)) this.preferences.ignoredWords.push(key);
    delete this.preferences.dictionary[key];
    await this.saveData(this.preferences);
    this.refreshPreferencesViews();
    new Notice("PT Tutor: ignoring " + word);
  }
  async addDictionaryEntry(word: string, replacement: string) {
    const key = norm(word), value = replacement.trim();
    if (!key || key.length > 100 || !value || value.length > 100) return;
    this.preferences.dictionary[key] = value;
    this.preferences.ignoredWords = this.preferences.ignoredWords.filter(w => w !== key);
    await this.saveData(this.preferences);
    this.refreshPreferencesViews();
    new Notice("PT Tutor: saved " + word + " → " + value);
  }
  async recordApproved(item: Suggestion) {
    if (!this.preferences.historyEnabled || item.original === item.replacement) return;
    const original = item.original.slice(0, 100), replacement = item.replacement.slice(0, 100);
    const history = this.preferences.history;
    const entry = history.find(row => row.original === original &&
      row.replacement === replacement && row.category === item.category);
    if (entry) {
      entry.count++;
      entry.lastApplied = new Date().toISOString();
    } else {
      history.unshift({ original, replacement, category: item.category,
        count: 1, lastApplied: new Date().toISOString() });
      if (history.length > 200) history.length = 200;
    }
    await this.saveData(this.preferences);
  }
  async onload() {
    const saved = await this.loadData() as Partial<TutorPreferences> | null;
    this.preferences = {
      ignoredWords: Array.isArray(saved?.ignoredWords) ? saved.ignoredWords.filter((w): w is string => typeof w === "string") : [],
      dictionary: saved?.dictionary && typeof saved.dictionary === "object" && !Array.isArray(saved.dictionary) ? saved.dictionary : {},
      historyEnabled: saved?.historyEnabled === true,
      history: Array.isArray(saved?.history) ? saved.history.filter((entry): entry is CorrectionRecord =>
        !!entry && typeof entry.original === "string" && typeof entry.replacement === "string" &&
        typeof entry.category === "string" && typeof entry.count === "number" &&
        Number.isSafeInteger(entry.count) && entry.count > 0 &&
        typeof entry.lastApplied === "string").slice(0, 200) : []
    };
    this.tutorExtension = createEditorExtension(this.app, this.preferences,
      word => this.addIgnored(word), (word, replacement) => this.addDictionaryEntry(word, replacement),
      item => this.recordApproved(item));
    const tutorExtension = this.tutorExtension;
    this.addSettingTab(new TutorDictionaryTab(this.app, this));
    this.registerEditorExtension([suggestionField, tutorExtension]);
    this.addCommand({
      id: "scan-entire-note",
      name: "Scan entire current note (up to 40 text lines)",
      callback: () => {
        new Notice("PT Tutor: scan command invoked (v2).");
        const cm = (this.app.workspace.activeEditor?.editor as unknown as { cm?: EditorView } | undefined)?.cm;
        const tutor = cm?.plugin(tutorExtension);
        if (!tutor) {
          new Notice("PT Tutor: no active CodeMirror editor; open note in editing mode.");
          return;
        }
        new Notice("PT Tutor: starting full note scan...");
        tutor.scanCurrentNote(true);
      }
    });
    this.addCommand({
      id: "diagnose-current-note",
      name: "Diagnose correction on current note",
      callback: async () => {
        const text = this.app.workspace.activeEditor?.editor?.getValue().trim().slice(-450);
        if (!text) { new Notice("PT Tutor: open a nonempty note first."); return; }
        try {
          const response = await requestUrl({
            url: "http://127.0.0.1:8765/v1/analyze", method: "POST",
            contentType: "application/json",
            body: JSON.stringify({ text, mode: "sentence" }), throw: false
          });
          if (response.status !== 200) {
            new Notice(`PT Tutor diagnostic: API HTTP ${response.status}`);
            return;
          }
          const edits = (response.json as Analysis).suggestions;
          new Notice(`PT Tutor diagnostic: ${Array.isArray(edits) ? edits.length : 0} grammar suggestion(s) returned.`);
        } catch (error) {
          new Notice(`PT Tutor diagnostic: cannot connect to local engine (${String(error).slice(0, 90)}).`);
        }
      }
    });
    this.addCommand({
      id: "check-local-engine",
      name: "Check local engine connection",
      callback: async () => {
        try {
          const response = await requestUrl({
            url: "http://127.0.0.1:8765/health",
            method: "GET",
            throw: false
          });
          if (response.status === 200 && response.json?.status === "ok") {
            new Notice(`PT English Tutor: engine connected (${response.json.model})`);
          } else {
            new Notice("PT English Tutor: engine unavailable. Check local backend.");
          }
        } catch {
          new Notice("PT English Tutor: engine offline. Start the local backend.");
        }
      }
    });
    new Notice("PT English Tutor loaded");
  }
}
 

class TutorDictionaryTab extends PluginSettingTab {
  constructor(app: App, private tutor: PTEnglishTutor) { super(app, tutor); }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "PT English Tutor — Personal Dictionary" });
    containerEl.createEl("p", { text: "Stored locally in this vault. No note text is saved." });
    let ignored = "";
    new Setting(containerEl).setName("Always ignore a word")
      .addText(text => text.setPlaceholder("e.g. Ollama").onChange(v => { ignored = v; }))
      .addButton(button => button.setButtonText("Add").onClick(async () => {
        await this.tutor.addIgnored(ignored); this.display();
      }));
    for (const word of this.tutor.preferences.ignoredWords) {
      new Setting(containerEl).setName(word)
        .addButton(button => button.setButtonText("Remove").onClick(async () => {
          this.tutor.preferences.ignoredWords = this.tutor.preferences.ignoredWords.filter(w => w !== word);
          await this.tutor.saveData(this.tutor.preferences);
          this.tutor.refreshPreferencesViews(); this.display();
        }));
    }
    containerEl.createEl("h3", { text: "Preferred corrections" });
    let word = "", translation = "";
    new Setting(containerEl).setName("Original word")
      .addText(text => text.setPlaceholder("e.g. sistema").onChange(v => { word = v; }));
    new Setting(containerEl).setName("Preferred English")
      .addText(text => text.setPlaceholder("e.g. system").onChange(v => { translation = v; }))
      .addButton(button => button.setButtonText("Save").onClick(async () => {
        await this.tutor.addDictionaryEntry(word, translation); this.display();
      }));
    for (const [term, replacement] of Object.entries(this.tutor.preferences.dictionary)) {
      new Setting(containerEl).setName(term + " → " + replacement)
        .addButton(button => button.setButtonText("Remove").onClick(async () => {
          delete this.tutor.preferences.dictionary[term];
          await this.tutor.saveData(this.tutor.preferences);
          this.tutor.refreshPreferencesViews(); this.display();
        }));
    }
    containerEl.createEl("h3", { text: "Approved corrections history" });
    containerEl.createEl("p", { text: "Optional and local. Only records corrections you explicitly apply; never automatically changes any note." });
    new Setting(containerEl).setName("Save approved corrections")
      .setDesc("Off by default. Stores individual corrected terms, categories and counts, never complete notes.")
      .addToggle(toggle => toggle.setValue(this.tutor.preferences.historyEnabled).onChange(async value => {
        this.tutor.preferences.historyEnabled = value;
        await this.tutor.saveData(this.tutor.preferences);
        this.display();
      }));
    new Setting(containerEl).setName("Clear correction history")
      .setDesc("Permanently delete all locally recorded corrections.")
      .addButton(button => button.setButtonText("Clear history").setWarning().onClick(async () => {
        this.tutor.preferences.history = [];
        await this.tutor.saveData(this.tutor.preferences);
        this.display();
      }));
    const sorted = [...this.tutor.preferences.history]
      .sort((a, b) => b.count - a.count);
    const practiceCards = sorted.filter(entry => entry.original !== entry.replacement);
    new Setting(containerEl).setName("Practice recurring mistakes")
      .setDesc("Practice corrections from your local approved history; never changes note content.")
      .addButton(button => button.setButtonText("Start practice").onClick(() => {
        if (!practiceCards.length) { new Notice("PT Tutor: no corrections to practice."); return; }
        // Round-robin categories so one kind of mistake does not dominate a session.
        const shuffle = <T>(items: T[]): T[] => {
          const result = [...items];
          for (let i = result.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [result[i], result[j]] = [result[j], result[i]];
          }
          return result;
        };
        const groups = shuffle(["translation", "spelling", "grammar"]).map(category =>
          shuffle(practiceCards.filter(card => card.category === category)));
        const mixed: CorrectionRecord[] = [];
        while (groups.some(group => group.length)) {
          for (const group of groups) {
            const card = group.shift();
            if (card) mixed.push(card);
          }
        }
        new TutorPracticeModal(this.app, mixed).open();
      }));
    containerEl.createEl("p", { text: sorted.length + " distinct approved corrections." });
    for (const entry of sorted) {
      new Setting(containerEl).setName(entry.original + " → " + entry.replacement)
        .setDesc(entry.category + " · applied " + entry.count + (entry.count === 1 ? " time" : " times") +
          " · last " + new Date(entry.lastApplied).toLocaleDateString())
        .addButton(button => button.setButtonText("Mark incorrect").onClick(async () => {
          this.tutor.preferences.history = this.tutor.preferences.history.filter(row => row !== entry);
          await this.tutor.saveData(this.tutor.preferences);
          this.display();
        }));
    }
  }
}

class TutorPracticeModal extends Modal {
  private index = 0;
  private correct = 0;
  private answerRevealed = false;
  constructor(app: App, private cards: CorrectionRecord[]) { super(app); }
  onOpen() { this.showCard(); }
  private showCard() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl("h3", { text: "PT English Tutor — Vocabulary Practice" });
    if (this.index >= this.cards.length) {
      contentEl.createEl("p", { text: "Finished! " + this.correct + "/" + this.cards.length + " correct." });
      contentEl.createEl("button", { text: "Close" }).addEventListener("click", () => this.close());
      return;
    }
    const card = this.cards[this.index];
    contentEl.createEl("p", { text: "Card " + (this.index + 1) + "/" + this.cards.length +
      " · " + card.category + " · previously corrected " + card.count + " time(s)" });
    if (card.category === "grammar") {
      // A new, synthetic sentence is built at practice time; no note text is stored.
      const verbHint = card.original.length <= card.replacement.length ? card.original : card.replacement;
      contentEl.createEl("p", { text: "Complete the sentence using the verb in parentheses. Change its form as needed:" });
      contentEl.createEl("p", { text: grammarPracticeSentence(card) });
      contentEl.createEl("p", { text: "Verb: (" + verbHint + ")" });
    } else if (card.category === "translation") {
      contentEl.createEl("p", { text: "Translate into English: " + card.original });
    } else {
      contentEl.createEl("p", { text: "Correct the spelling: " + card.original });
    }
    const field = contentEl.createEl("input", { type: "text", attr: { "aria-label": "Your correction" } });
    field.style.width = "100%";
    const feedback = contentEl.createEl("p");
    feedback.setAttribute("role", "status");
    feedback.setAttribute("aria-live", "polite");
    const next = contentEl.createEl("button", { text: "Next card" });
    next.disabled = true;
    this.answerRevealed = false;
    const checkButton = contentEl.createEl("button", { text: "Check answer" });
    const checkAnswer = () => {
      if (this.answerRevealed) return;
      this.answerRevealed = true;
      const right = field.value.trim().toLocaleLowerCase() === card.replacement.trim().toLocaleLowerCase();
      if (right) this.correct++;
      feedback.setText((right ? "Correct! " : "Review: ") + card.original + " → " + card.replacement);
      next.disabled = false;
      field.disabled = true;
      checkButton.disabled = true;
      next.focus();
    };
    checkButton.addEventListener("click", checkAnswer);
    field.addEventListener("keydown", event => {
      if (event.key === "Enter") { event.preventDefault(); checkAnswer(); }
    });
    next.addEventListener("click", () => { this.index++; this.showCard(); });
    field.focus();
  }
}

// Synthetic grammar context, never extracted from notes or saved to history.
// Curated complements keep each sentence meaningful instead of testing verb forms alone.
function grammarPracticeSentence(card: CorrectionRecord): string {
  const verb = card.replacement.toLowerCase();
  const singular = (verb.endsWith("s") && !verb.endsWith("ss")) || verb === "has" || verb === "does";
  const subject = singular ? "The application" : "The applications";
  const stems: Record<string, string> = {
    depend: "___ on reliable data.", depends: "___ on reliable data.",
    have: "___ enough memory.", has: "___ enough memory.",
    do: "___ the work correctly.", does: "___ the work correctly.",
    go: "___ online every morning.", goes: "___ online every morning.",
    use: "___ local tools.", uses: "___ local tools.",
    generate: "___ useful reports.", generates: "___ useful reports.",
    process: "___ incoming requests.", processes: "___ incoming requests.",
    run: "___ without errors.", runs: "___ without errors.",
    work: "___ reliably.", works: "___ reliably.",
    need: "___ regular updates.", needs: "___ regular updates.",
    receive: "___ new information.", receives: "___ new information.",
    send: "___ notifications.", sends: "___ notifications.",
    collect: "___ data.", collects: "___ data.",
    contain: "___ several modules.", contains: "___ several modules.",
    require: "___ maintenance.", requires: "___ maintenance.",
    write: "___ reports.", writes: "___ reports.",
    test: "___ the changes.", tests: "___ the changes.",
    improve: "___ performance.", improves: "___ performance.",
    organize: "___ tasks.", organizes: "___ tasks."
  };
  // Unrecognized grammar forms still get a sentence with a meaningful context.
  return subject + " " + (stems[verb] ?? "___ as the project evolves.");
}
