// Color mapping for node encodings — "Morning Mist" light theme.
// Severity set and risk ramp validated with the dataviz palette validator
// against the mist surface (#f4f2fa): the three active severities pass the
// lightness band, chroma floor, and CVD-separation checks; "none" and the
// ramp's light end deliberately recede toward the surface. Severity is never
// color-alone (legend text, tooltips, inspector readouts).

export const SEVERITY_COLORS = ["#d8d6e8", "#d4a24a", "#cf7256", "#a94a74"];
export const SEVERITY_NAMES = ["none", "minor", "moderate", "major"];

// Sequential ramp for model risk on the light surface: near-zero fades into
// the mist, high risk deepens to violet. Mirrors --seq-* in styles.css.
const RISK_STOPS: [number, string][] = [
  [0.0, "#eceafa"],
  [0.33, "#c3c1f0"],
  [0.66, "#8b93e6"],
  [1.0, "#5b5cb8"],
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
