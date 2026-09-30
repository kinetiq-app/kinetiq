// sw.js — offline shell for the Kinetiq v4 PWA.
//
// STRATEGY: network-first for the app shell, cache only as an offline fallback.
//
// The previous version was cache-first with a fixed cache name, which pinned
// app.js and config.js in the browser forever: every redeploy was invisible to
// anyone who had already opened the site once, and a stale config.js kept
// pointing at http://localhost:8000 long after the API URL had changed. For an
// app that redeploys often and whose API URL lives IN the shell, correctness
// beats the few milliseconds cache-first saves.
//
// Offline still works — the cache is written on every successful fetch and
// served whenever the network fails.
//
// CACHE is versioned. Bump it whenever the shell's caching behaviour changes;
// `activate` deletes every cache that is not the current one.
const CACHE = "kinetiq-v6-shell-v5";

const SHELL = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./segments.js",
  "./config.js",
  "./severities.json",
  "./manifest.json",
  "./kinetiq-logo.jpg",
];

self.addEventListener("install", (e) => {
  // Pre-seed the offline fallback. Individual failures must not abort the whole
  // install, so each is caught separately.
  e.waitUntil(
    caches
      .open(CACHE)
      .then((c) =>
        Promise.all(
          SHELL.map((u) =>
            c.add(new Request(u, { cache: "reload" })).catch(() => {})
          )
        )
      )
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  // Never intercept the CDN (pose model/wasm) or the detector API. Those must
  // reach the network on their own terms; caching an assess response would be
  // actively wrong.
  if (url.origin !== self.location.origin) return;
  if (url.pathname === "/health" || url.pathname.startsWith("/prototype/")) return;

  e.respondWith(
    fetch(req)
      .then((res) => {
        // Refresh the offline copy on every good response.
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() =>
        // Offline: fall back to whatever we last saw, then to the shell.
        caches.match(req).then((hit) => hit || caches.match("./index.html"))
      )
  );
});
