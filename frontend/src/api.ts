import type { ModelStatus, NetworkResponse } from "./types";

export async function fetchNetwork(): Promise<NetworkResponse> {
  const res = await fetch("/api/network");
  if (!res.ok) throw new Error(`GET /api/network -> ${res.status}`);
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch("/api/model");
  if (!res.ok) throw new Error(`GET /api/model -> ${res.status}`);
  return res.json();
}

export async function startTraining(quality: "fast" | "full"): Promise<void> {
  await fetch("/api/model/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quality }),
  });
}
