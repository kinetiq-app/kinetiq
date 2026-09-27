// Cloudflare Worker entry point for kinetiq PWA with static assets
// Proxies backend detector API endpoints (/health and /prototype/*) to Render,
// and serves static assets for all client PWA routes.
export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CORS preflight directly at the edge
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    // Proxy API routes (/health and /prototype/*) to the Render detector backend
    if (url.pathname === "/health" || url.pathname.startsWith("/prototype/")) {
      const targetUrl = new URL(url.pathname + url.search, "https://kinetiq-v5-api.onrender.com");

      const headers = new Headers(request.headers);
      headers.set("Host", "kinetiq-v5-api.onrender.com");
      headers.set("Origin", "https://kinetiq-v5-pwa.onrender.com");

      const backendRequest = new Request(targetUrl.toString(), {
        method: request.method,
        headers: headers,
        body: request.body,
        redirect: "follow",
      });

      try {
        const response = await fetch(backendRequest);
        const newHeaders = new Headers(response.headers);
        newHeaders.set("Access-Control-Allow-Origin": "*");
        newHeaders.set("Access-Control-Allow-Methods": "GET, POST, OPTIONS");
        newHeaders.set("Access-Control-Allow-Headers": "Content-Type, Authorization");

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: newHeaders,
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: "Backend proxy error", detail: String(err) }), {
          status: 502,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        });
      }
    }

    // Fallback: serve static assets from the frontend directory
    return env.ASSETS.fetch(request);
  },
};
