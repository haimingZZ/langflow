# Backend Component Strings — i18n Architecture

## The Problem

Component names, descriptions, and field labels shown in the Langflow UI (sidebar, node headers, tooltips) come from **Python backend metadata**, not the frontend. Examples:

- `display_name = "Chat Input"` — shown in sidebar list and node title
- `description = "Get chat inputs from the Playground."` — shown in tooltips
- Input/output `display_name` fields — shown in node template fields

Because these strings originate in Python and are served via the REST API, the frontend's `t()` approach cannot translate them — by the time React renders them, they're just API response strings.

---

## Implemented Architecture (Option B: Accept-Language)

**Decision:** Backend owns its own locale files, serves translated metadata via `Accept-Language` header. Frontend sends the header; backend returns already-translated strings. No frontend key lookups for component strings.

### Data Flow

```
User selects language in UI
  → LanguageSelector calls i18n.changeLanguage() + clears types cache
  → Every API request includes Accept-Language: fr
  → LocaleMiddleware parses header → request.state.locale = "fr"
  → /api/v1/all reads request.state.locale
  → Cached English dict is copied + display_names substituted from fr.json
  → Frontend renders translated strings as-is — no t() lookups needed
```

### Backend Locale Files

```
src/backend/base/langflow/locales/
  en.json       ← source strings (GP-compatible, flat key-value)
  fr.json       ← GP-produced
  es.json
  ja.json
  de.json
  pt.json
  zh-Hans.json
```

**Flat key format** (matches frontend en.json convention, required for GP):
```json
{
  "components.ChatInput.display_name": "Chat Input",
  "components.ChatInput.description": "Get chat inputs from the Playground.",
  "components.ChatInput.inputs.input_value.display_name": "Input Text",
  "components.ChatInput.outputs.message.display_name": "Chat Message"
}
```

4,748 keys across 359 components (Tier 1 + Tier 2 display_names).

### GP Bundle

Backend strings use a **separate GP bundle** (`langflow-ui-backend`) distinct from the frontend bundle. This keeps submission clean per layer and lets the backend serve any API consumer independently.

---

## Implemented Files

| File | Role |
|------|------|
| `src/backend/base/langflow/utils/i18n.py` | Loads locale JSON files; `translate(key, locale, default)` + `translate_component_dict()` |
| `src/backend/base/langflow/locales/en.json` | Source strings, bootstrapped by extraction script |
| `src/backend/base/langflow/locales/{fr,es,ja,de,pt,zh-Hans}.json` | GP-translated files |
| `src/backend/base/langflow/main.py` | `set_locale` middleware — parses Accept-Language, sets `request.state.locale` |
| `src/backend/base/langflow/api/v1/endpoints.py` | `get_all` reads `request.state.locale`, calls `translate_component_dict()` for non-en |
| `src/frontend/src/customization/hooks/use-custom-api-headers.ts` | Sends `Accept-Language: <current language>` on every request |
| `src/frontend/src/components/core/appHeaderComponent/components/LanguageSelector/index.tsx` | On language change: clears `useTypesStore` + invalidates `["useGetTypes"]` React Query cache |
| `scripts/gp/extract_backend_strings.py` | Walks `lfx.components`, extracts Tier 1+2 strings to `locales/en.json` |
| `scripts/gp/upload_backend_strings.py` | Uploads `en.json` to GP `langflow-ui-backend` bundle |
| `scripts/gp/download_backend_translations.py` | Downloads translated files from GP into `locales/` |

---

## Translation Scope

### Tier 1 (implemented)
- Component `display_name` and `description`

### Tier 2 (implemented)
- Input and output field `display_name`

### Phase 2 (future — not yet implemented)
- Field `info` tooltip text
- Field `placeholder` text
- `options` arrays — value-based keys (e.g. `components.X.inputs.Y.options.Machine`); product identifiers marked "do not translate" in GP

No infrastructure changes are needed for Phase 2 — the extraction script and middleware are already in place. Add more keys to `en.json` and extend `translate_component_dict()` to substitute the additional fields.

---

## Operational Workflow

### Updating source strings (when components change)

```bash
# From repo root with backend virtualenv active:
python scripts/gp/extract_backend_strings.py

# CI check (exits 1 if en.json is out of sync):
python scripts/gp/extract_backend_strings.py --check
```

### Uploading to GP

```bash
cd scripts/gp
python upload_backend_strings.py
```

### Downloading translations from GP

```bash
cd scripts/gp
python download_backend_translations.py
```

---

## Known Limitations / Future Work

### Canvas node reconciliation (not yet fixed)

**Problem:** When a user changes language, existing nodes already placed on the canvas do **not** update their displayed strings. Only newly added nodes show the new language.

**Root cause:** When a node is dragged onto the canvas, its template data is deep-copied into the flow's node state (a snapshot). This snapshot is disconnected from `useTypesStore`. Re-fetching `/api/v1/all` updates the store (and sidebar), but does not patch the snapshots of existing nodes.

**Impact:** Low — only affects open flows during the same session. A page reload or re-adding nodes shows the correct language.

**Proposed fix (Phase 2):**
- After `queryClient.invalidateQueries(["useGetTypes"])` succeeds, iterate all nodes in `useFlowStore` and update their `data.node.template[*].display_name` from the freshly fetched types dict
- Key mapping: `node.data.type` → component key in all_types dict
- Requires care not to overwrite user-modified field values

### English fast path

`translate_component_dict()` is only called when `locale != "en"`. English requests return the cached dict directly with no copy overhead.

### Locale fallback chain

`translate(key, locale, default)` falls back: requested locale → "en" dict → raw Python string. Unknown locales silently degrade to English.
