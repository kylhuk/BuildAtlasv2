# UI Environment

## Environment files
The UI consumes Vite environment variables from `.env*` files in `ui/`. Copy `ui/.env.example` to `.env.development.local` (or `.env`) when you want to override defaults for your machine.

## Convention
- Only keys prefixed with `VITE_` are embedded into the client bundle, so keep secrets server-only.
- `VITE_BACKEND_URL` should point to the backend dev server (e.g. `http://localhost:8000`).
- `VITE_API_TIMEOUT_MS` and `VITE_APP_TITLE` are optional helpers for fetch timeouts or display strings.

## Example file
See `ui/.env.example` for a propagated template. Update it with real endpoints before you run `npm run dev`.
