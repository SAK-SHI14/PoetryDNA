const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    let message = "The archive refused the request.";
    try {
      const payload = await response.json();
      message = payload.detail ?? message;
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }

  return response.json();
}

export function predictPoet(text) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export function explainPoetry(text, include = ["tokens", "distilbert", "sbert"]) {
  return request("/explain", {
    method: "POST",
    body: JSON.stringify({ text, include }),
  });
}

export function getCorpusEmbeddings() {
  return request("/corpus-embeddings");
}

export function getDatasetStats() {
  return request("/dataset-stats");
}

export function explainNlp(text) {
  return request("/explain", {
    method: "POST",
    body: JSON.stringify({ text, include: ["nlp"] }),
  });
}
