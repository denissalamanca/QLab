// Typed client for the AFML control-plane API (Blueprint §11.1).
// Mirrors the Pydantic schemas in src/afml/control_plane/schemas.py.

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api/v1";

export interface Strategy {
  experiment_id: string;
  asset: string;
  algorithmic_family: string;
  agent_version: string;
  timestamp: string;
  num_events_triggered: number;
  orthogonality_score: number | null;
  brain_1_recall: number | null;
  brain_2_log_loss: number | null;
  pbo: number | null;
  dsr: number | null;
  is_deployed: boolean;
  status: string;
}

export interface ApproveRequest {
  experiment_id: string;
  timestamp_ms: number;
  signed_token: string;
  totp_code: string;
}

export interface ApproveResponse {
  experiment_id: string;
  deployed: boolean;
  message: string;
}

export interface FlattenRequest {
  nonce: string;
  timestamp_ms: number;
  signed_token: string;
  reason: string;
  reference_prices?: Record<string, number>;
}

export interface FlattenFill {
  asset: string;
  side: string;
  size: number;
  fill_price: number;
  status: string;
}

export interface FlattenResponse {
  flattened: boolean;
  n_positions_closed: number;
  reason: string;
  fills: FlattenFill[];
}

// Canonical messages the CEO signs (must match afml.crypto.signing). The
// millisecond timestamp is bound into the signature; the backend rejects it
// outside a ±60 s window (audit V1 anti-replay).
export const approvalMessage = (experimentId: string, timestampMs: number): string =>
  `afml:approve:${experimentId}:${timestampMs}`;
export const flattenMessage = (nonce: string, timestampMs: number): string =>
  `afml:flatten:${nonce}:${timestampMs}`;

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = (await resp.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // non-JSON error body — keep the status text.
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}

export const api = {
  listStrategies: (): Promise<Strategy[]> => request<Strategy[]>("/registry/strategies"),

  approve: (body: ApproveRequest): Promise<ApproveResponse> =>
    request<ApproveResponse>("/execution/approve", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  flatten: (body: FlattenRequest): Promise<FlattenResponse> =>
    request<FlattenResponse>("/emergency/flatten", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
