import type { CustomNodeDef, ModelStatus, NetworkResponse } from "./types";

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

async function throwApiErrors(res: Response): Promise<never> {
  let detail = `${res.status}`;
  try {
    const body = await res.json();
    if (Array.isArray(body.errors)) detail = body.errors.join("; ");
  } catch {
    /* non-JSON error body */
  }
  throw new Error(detail);
}

export async function addCustomNode(
  def: Partial<CustomNodeDef>,
): Promise<{ node: CustomNodeDef & { id: number }; model: ModelStatus }> {
  const res = await fetch("/api/network/custom", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(def),
  });
  if (!res.ok) await throwApiErrors(res);
  return res.json();
}

export async function removeCustomNode(name: string): Promise<void> {
  const res = await fetch(
    `/api/network/custom/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
  if (!res.ok) await throwApiErrors(res);
}
