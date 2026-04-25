---
name: Cloudflare Worker
description: Patterns for building Cloudflare Workers with Wrangler, Hono, and TypeScript
---

# Cloudflare Workers + Wrangler + Hono

## Project Setup
```bash
bun create hono my-worker --template cloudflare-workers
cd my-worker
bun install
```

## wrangler.jsonc (preferred over wrangler.toml)
```jsonc
{
  "name": "my-worker",
  "main": "src/index.ts",
  "compatibility_date": "2026-04-01",
  "compatibility_flags": ["nodejs_compat"],
  // Bindings
  "kv_namespaces": [{ "binding": "KV", "id": "..." }],
  "d1_databases": [{ "binding": "DB", "database_name": "...", "database_id": "..." }],
  "r2_buckets": [{ "binding": "BUCKET", "bucket_name": "..." }]
}
```

## Hono Pattern
```ts
import { Hono } from "hono";
import { cors } from "hono/cors";

type Bindings = {
  KV: KVNamespace;
  DB: D1Database;
  API_KEY: string; // secret
};

const app = new Hono<{ Bindings: Bindings }>();

app.use("*", cors());

app.get("/", (c) => c.json({ status: "ok" }));

app.get("/items/:id", async (c) => {
  const id = c.req.param("id");
  const result = await c.env.DB.prepare("SELECT * FROM items WHERE id = ?")
    .bind(id)
    .first();
  if (!result) return c.json({ error: "not found" }, 404);
  return c.json(result);
});

export default app;
```

## Key Conventions

### Bindings & Secrets
- Define types for all bindings in `Bindings` type
- Secrets set via `wrangler secret put SECRET_NAME`
- Access via `c.env.SECRET_NAME` in Hono
- Never hardcode secrets

### D1 (SQLite)
- Use parameterized queries (`.bind()`) — never string interpolation
- Migrations in `migrations/` dir, run with `wrangler d1 migrations apply`
- `first()` for single row, `all()` for multiple

### KV
- `await c.env.KV.get(key)` / `.put(key, value)`
- JSON: `get(key, { type: "json" })` / `put(key, JSON.stringify(val))`
- TTL: `put(key, val, { expirationTtl: 3600 })`

### Error Handling
```ts
app.onError((err, c) => {
  console.error(err);
  return c.json({ error: "Internal Server Error" }, 500);
});
```

### Development
```bash
wrangler dev              # Local dev server
wrangler deploy           # Deploy to production
wrangler tail             # Stream live logs
wrangler d1 execute DB --local --command "SELECT * FROM items"
```

## Do NOT
- Use `addEventListener("fetch", ...)` — use Hono or module syntax
- Import Node.js built-ins without `nodejs_compat` flag
- Use `fetch()` without error handling (Workers have 30s CPU time limit)
- Store large objects in KV (use R2 for >25MB)
- Use `JSON.parse(await kv.get(key))` — use `get(key, { type: "json" })`
