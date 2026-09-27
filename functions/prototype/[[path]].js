export async function onRequest(context) {
  const { request, params } = context;
  const path = Array.isArray(params.path) ? params.path.join("/") : params.path || "";
  const url = new URL(request.url);
  const target = `https://kinetiq-v5-api.onrender.com/prototype/${path}${url.search}`;

  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  }

  const headers = new Headers(request.headers);
  headers.set("Host", "kinetiq-v5-api.onrender.com");
  headers.set("Origin", "https://kinetiq-v5-pwa.onrender.com");

  try {
    const res = await fetch(target, {
      method: request.method,
      headers,
      body: request.body,
    });
    const newHeaders = new Headers(res.headers);
    newHeaders.set("Access-Control-Allow-Origin", "*");
    return new Response(res.body, { status: res.status, headers: newHeaders });
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 502,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  }
}
