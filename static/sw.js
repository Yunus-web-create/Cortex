/* Cortex service worker — makes the app installable and usable on a flaky phone
   connection. It caches the *shell* only: never API responses, never media, never
   a streamed answer, so the app can never show you stale data. */
const VERSION = "cortex-v2";
const SHELL = [
  "/",
  "/static/app.js",
  "/static/styles.css",
  "/manifest.webmanifest",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(VERSION)
      .then((cache) => cache.addAll(SHELL).catch(() => {}))   // a miss must not block install
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;                       // never intercept a chat POST

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;            // leave third parties alone
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/media/")) return;

  // network first, so a deploy or a fix shows up immediately
  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response && response.ok && response.status === 200) {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put(request, copy)).catch(() => {});
        }
        return response;
      })
      .catch(async () => {
        const hit = await caches.match(request);
        if (hit) return hit;
        if (request.mode === "navigate") {
          const shell = await caches.match("/");
          if (shell) return shell;
        }
        return new Response("Cortex is offline. Reconnect and try again.", {
          status: 503, headers: { "Content-Type": "text/plain; charset=utf-8" },
        });
      })
  );
});
