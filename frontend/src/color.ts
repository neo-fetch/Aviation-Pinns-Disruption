// Color mapping for node encodings, using the palette tokens in styles.css.

export const SEVERITY_COLORS = ["#383835", "#fab219", "#ec835a", "#d03b3b"];
export const SEVERITY_NAMES = ["none", "minor", "moderate", "major"];

// Sequential ramp for model risk on the dark surface: near-zero recedes
// toward the surface, high risk is the brightest blue step.
const RISK_STOPS: [number, string][] = [
  [0.0, "#222221"],
  [0.33, "#184f95"],
  [0.66, "#3987e5"],
  [1.0, "#86b6ef"],
];

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function riskColor(v: number): string {
  const x = Math.max(0, Math.min(1, v));
  for (let i = 1; i < RISK_STOPS.length; i++) {
    const [x1, c1] = RISK_STOPS[i - 1];
    const [x2, c2] = RISK_STOPS[i];
    if (x <= x2) {
      const t = (x - x1) / (x2 - x1);
      const a = hexToRgb(c1);
      const b = hexToRgb(c2);
      const mix = a.map((av, j) => Math.round(av + (b[j] - av) * t));
      return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
    }
  }
  return RISK_STOPS[RISK_STOPS.length - 1][1];
}

export const TIER_LABELS: Record<string, string> = {
  raw: "Raw materials",
  tier2: "Tier-2 suppliers",
  tier1: "Tier-1 systems",
  plant: "Airbus plants",
  fal: "Final assembly",
  customer: "Deliveries",
};
