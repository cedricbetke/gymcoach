import {
  LlmError,
  type LlmCompletionRequest,
  type LlmCompletionResult,
  type LlmProvider,
} from './types';

// KI:connect – OpenAI Chat Completions-compatible endpoint hosted by the
// state of NRW. Configured via three env vars (see .env.example).
const DEFAULT_BASE_URL = 'https://chat.kiconnect.nrw/api/v1';
const DEFAULT_MODEL = 'inferenz-gpt-oss-120b';
const DEFAULT_MAX_TOKENS = 8000;

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, '');
}

interface ChatCompletionResponse {
  model?: string;
  choices?: Array<{ message?: { role: string; content: string } }>;
  error?: { message: string; code?: number | string };
}

// Parses one SSE line from a Chat Completions streaming response. Returns the
// text delta, or null for keep-alives, the [DONE] sentinel, and non-data lines.
export function extractKiConnectDelta(line: string): string | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith('data:')) return null;
  const payload = trimmed.slice(5).trim();
  if (payload === '' || payload === '[DONE]') return null;
  try {
    const json = JSON.parse(payload) as {
      choices?: Array<{ delta?: { content?: string } }>;
    };
    return json.choices?.[0]?.delta?.content ?? null;
  } catch {
    return null;
  }
}

export class KiConnectProvider implements LlmProvider {
  readonly id = 'kiconnect' as const;
  readonly label = 'KI:connect';
  readonly apiKeyEnvVar = 'KICONNECT_API_KEY';
  readonly model: string;
  private readonly apiKey: string | undefined;
  private readonly baseUrl: string;

  constructor() {
    this.apiKey = process.env.KICONNECT_API_KEY?.trim();
    this.baseUrl = normalizeBaseUrl(
      process.env.KICONNECT_BASE_URL?.trim() || DEFAULT_BASE_URL,
    );
    this.model = process.env.KICONNECT_MODEL?.trim() || DEFAULT_MODEL;
  }

  isConfigured(): boolean {
    return !!this.apiKey;
  }

  async complete(req: LlmCompletionRequest): Promise<LlmCompletionResult> {
    if (!this.apiKey) {
      throw new LlmError(503, 'KICONNECT_API_KEY is not configured.');
    }

    const body = {
      model: this.model,
      messages: [
        { role: 'system', content: req.system },
        ...req.messages.map((m) => ({ role: m.role, content: m.content })),
      ],
      ...(req.temperature != null ? { temperature: req.temperature } : {}),
      max_tokens: req.maxTokens ?? DEFAULT_MAX_TOKENS,
    };

    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(body),
      });
    } catch (err) {
      throw new LlmError(
        502,
        `Network failure to KI:connect: ${err instanceof Error ? err.message : 'unknown'}`,
      );
    }

    if (!res.ok) {
      const text = await res.text();
      throw new LlmError(res.status, `KI:connect ${res.status}: ${text.slice(0, 500)}`);
    }

    const json = (await res.json()) as ChatCompletionResponse;
    if (json.error) {
      throw new LlmError(502, `KI:connect: ${json.error.message}`);
    }
    const text = json.choices?.[0]?.message?.content?.trim();
    if (!text) {
      throw new LlmError(502, 'Empty response from the coach.');
    }
    return { text, modelUsed: json.model ?? this.model };
  }

  async *stream(req: LlmCompletionRequest): AsyncIterable<string> {
    if (!this.apiKey) {
      throw new LlmError(503, 'KICONNECT_API_KEY is not configured.');
    }

    const body = {
      model: this.model,
      messages: [
        { role: 'system', content: req.system },
        ...req.messages.map((m) => ({ role: m.role, content: m.content })),
      ],
      ...(req.temperature != null ? { temperature: req.temperature } : {}),
      max_tokens: req.maxTokens ?? DEFAULT_MAX_TOKENS,
      stream: true,
    };

    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(body),
      });
    } catch (err) {
      throw new LlmError(
        502,
        `Network failure to KI:connect: ${err instanceof Error ? err.message : 'unknown'}`,
      );
    }

    if (!res.ok) {
      const text = await res.text();
      throw new LlmError(res.status, `KI:connect ${res.status}: ${text.slice(0, 500)}`);
    }
    if (!res.body) {
      throw new LlmError(502, 'KI:connect returned no response body.');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const delta = extractKiConnectDelta(line);
        if (delta) yield delta;
      }
    }

    // Flush any remaining bytes in the decoder and handle a final partial line.
    buffer += decoder.decode();
    const tail = extractKiConnectDelta(buffer);
    if (tail) yield tail;
  }
}
