export async function onRequest(context) {
  const { request } = context;
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  }
  const headers = new Headers(request.headers);
  headers.set("Host", "kinetiq-v5-api.onrender.com");
  headers.set("Origin", "https://kinetiq-v5-pwa.onrender.com");

  try {
    const res = await fetch("https://kinetiq-v5-api.onrender.com/health", {
      method: "GET",
      headers,
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
