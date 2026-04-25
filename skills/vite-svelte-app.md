---
name: Vite + Svelte App
description: Conventions for building web apps with Vite, Svelte 5, TypeScript, and Tailwind CSS
---

# Vite + Svelte 5 + TypeScript + Tailwind CSS

## Project Setup
```bash
bun create vite my-app --template svelte-ts
cd my-app
bun install
bun add -D tailwindcss @tailwindcss/vite
```

## Key Conventions

### Svelte 5 Runes (NOT legacy reactive syntax)
- Use `$state()` for reactive state, NOT `let x = 0`
- Use `$derived()` for computed values, NOT `$: x = ...`
- Use `$effect()` for side effects, NOT `$: { ... }`
- Use `$props()` for component props, NOT `export let`
- Use `$bindable()` for two-way bindable props
- Use `{#snippet}` blocks, NOT `<slot>`

### File Structure
```
src/
├── lib/
│   ├── components/    # Reusable UI components
│   ├── stores/        # Shared state ($state in .svelte.ts files)
│   ├── utils/         # Pure helper functions
│   └── types.ts       # Shared TypeScript types
├── routes/            # SvelteKit routes (if using SK)
├── app.css            # Tailwind imports
└── App.svelte         # Root component (Vite SPA)
```

### Component Pattern
```svelte
<script lang="ts">
  let { title, count = $bindable(0) }: { title: string; count: number } = $props();

  let doubled = $derived(count * 2);

  function increment() {
    count++;
  }
</script>

<button onclick={increment} class="px-4 py-2 bg-blue-500 text-white rounded">
  {title}: {count} (doubled: {doubled})
</button>
```

### Styling
- Use Tailwind utility classes directly in markup
- `app.css`: `@import "tailwindcss";`
- Scoped styles with `<style>` block when Tailwind isn't sufficient
- No CSS modules or CSS-in-JS

### State Management
- Component-local: `$state()` in `.svelte` files
- Shared: `$state()` in `.svelte.ts` files (importable stores)
- No external state libraries unless complexity demands it

### TypeScript
- Strict mode enabled
- Props typed inline with `$props()` or via interfaces
- Avoid `any` — use `unknown` and narrow

### Vite Config
```ts
import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [svelte(), tailwindcss()],
});
```

## Do NOT
- Use Svelte 4 syntax (`export let`, `$:`, `<slot>`)
- Use `class:` directive when Tailwind suffices
- Install `postcss` or `autoprefixer` (Tailwind v4 Vite plugin handles it)
- Use `onMount`/`onDestroy` when `$effect` works
