const CACHE='mei-data-check-v1';
const shell=['/','/style.css','/app.mjs','/catalog.mjs','/checks.mjs','/storage.mjs','/parse-worker.js','/mei-worker.mjs','/node-task.mjs','/vendor/xlsx.full.min.js','/sdk/wasm-abi.mjs','/assets/mei-model.json','/assets/tensors.bin','/assets/tokenizer.model','/assets/tool-index.json','/assets/runtime.wasm','/assets/receipt.json'];
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(shell)).then(()=>self.skipWaiting())));
self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));
self.addEventListener('fetch',e=>{const u=new URL(e.request.url);if(e.request.method!=='GET'||u.origin!==self.location.origin||u.pathname.startsWith('/api/'))return;e.respondWith(fetch(e.request).catch(()=>caches.match(e.request)));});
