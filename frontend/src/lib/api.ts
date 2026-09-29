/**
 * Typed client for the Constellation FastAPI backend.
 * Types mirror backend/constellation/api/schemas.py.
 */

const BASE = import.meta.env.VITE_API_URL ?? "";

export interface Song {
  id: number;
  title: string | null;
  artist: string | null;
  album: string | null;
  duration_s: number;
  source: string;
  source_id: string | null;
  audio_url: string;
  created_at: string | null;
}

export interface SongPage {
  items: Song[];
  total: number;
  limit: number;
  offset: number;
}

export interface IdentifyResult {
  match: boolean;
  song: Song | null;
  offset_s: number | null;
  confidence: number;
  aligned_matches: number;
  runner_up_matches: number;
  query: { duration_s: number; hashes: number; peaks: [number, number][] };
  timing: { decode_ms: number; fingerprint_ms: number; match_ms: number; total_ms: number };
}

export type SortField = "id" | "title" | "artist" | "duration_s";
export type SortOrder = "asc" | "desc";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, init);
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "Can't reach the server. Is the API running?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export function audioUrl(song: Song): string {
  return BASE + song.audio_url;
}

export function identify(blob: Blob, filename: string, signal?: AbortSignal): Promise<IdentifyResult> {
  const form = new FormData();
  form.append("file", blob, filename);
  return request("/identify", { method: "POST", body: form, signal });
}

export function listSongs(
  params: { q?: string; sort?: SortField; order?: SortOrder; limit?: number; offset?: number },
  signal?: AbortSignal,
): Promise<SongPage> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") qs.set(k, String(v));
  return request(`/songs?${qs}`, { signal });
}

export function randomSong(): Promise<Song> {
  return request("/songs/random");
}

export function health(): Promise<{ status: string; songs: number; fingerprint_version: string }> {
  return request("/health");
}
