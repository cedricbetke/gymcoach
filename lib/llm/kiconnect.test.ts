import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { KiConnectProvider, extractKiConnectDelta } from './kiconnect';
import { LlmError } from './types';

const ENV_KEYS = ['KICONNECT_API_KEY', 'KICONNECT_BASE_URL', 'KICONNECT_MODEL'] as const;
const savedEnv: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const key of ENV_KEYS) savedEnv[key] = process.env[key];
  for (const key of ENV_KEYS) delete process.env[key];
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
  vi.unstubAllGlobals();
});

describe('KiConnectProvider', () => {
  it('reports not configured and throws 503 without a key', async () => {
    const provider = new KiConnectProvider();
    expect(provider.isConfigured()).toBe(false);
    await expect(
      provider.complete({ system: 'S', messages: [{ role: 'user', content: 'hi' }] }),
    ).rejects.toMatchObject({ status: 503 });
  });

  it('uses the default model when KICONNECT_MODEL is unset', () => {
    process.env.KICONNECT_API_KEY = 'kc-key';
    const provider = new KiConnectProvider();
    expect(provider.model).toBe('inferenz-gpt-oss-120b');
  });

  it('respects the KICONNECT_MODEL override', () => {
    process.env.KICONNECT_API_KEY = 'kc-key';
    process.env.KICONNECT_MODEL = 'custom-model';
    const provider = new KiConnectProvider();
    expect(provider.model).toBe('custom-model');
  });

  it('sends the system message, forwards temperature, and parses the reply', async () => {
    process.env.KICONNECT_API_KEY = 'kc-key';
    process.env.KICONNECT_BASE_URL = 'https://chat.kiconnect.test/api/v1/';
    process.env.KICONNECT_MODEL = 'test-model';

    const fetchMock = vi.fn(
      async (_url: string, _init: RequestInit) =>
        new Response(
          JSON.stringify({
            model: 'test-model',
            choices: [{ message: { role: 'assistant', content: '  Hello coach  ' } }],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const provider = new KiConnectProvider();
    const result = await provider.complete({
      system: 'SYSTEM',
      messages: [{ role: 'user', content: 'payload' }],
      temperature: 0.5,
      maxTokens: 512,
    });

    expect(result).toEqual({ text: 'Hello coach', modelUsed: 'test-model' });

    const [url, init] = fetchMock.mock.calls[0]!;
    // Trailing slash in KICONNECT_BASE_URL must be normalized away.
    expect(url).toBe('https://chat.kiconnect.test/api/v1/chat/completions');
    expect((init?.headers as Record<string, string>).Authorization).toBe('Bearer kc-key');
    const body = JSON.parse(init?.body as string);
    expect(body.messages[0]).toEqual({ role: 'system', content: 'SYSTEM' });
    expect(body.messages[1]).toEqual({ role: 'user', content: 'payload' });
    expect(body.temperature).toBe(0.5);
    expect(body.max_tokens).toBe(512);
    expect(body.model).toBe('test-model');
    // API key must never appear in the request body.
    expect(JSON.stringify(body)).not.toContain('kc-key');
  });

  it('omits temperature when it is not provided', async () => {
    process.env.KICONNECT_API_KEY = 'kc-key';

    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            choices: [{ message: { role: 'assistant', content: 'ok' } }],
          }),
          { status: 200 },
        ),
      ),
    );

    const provider = new KiConnectProvider();
    await provider.complete({ system: 'S', messages: [{ role: 'user', content: 'x' }] });

    const [, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0]!;
    const body = JSON.parse(init.body as string);
    expect('temperature' in body).toBe(false);
  });

  it('maps an upstream HTTP error to LlmError with the upstream status', async () => {
    process.env.KICONNECT_API_KEY = 'kc-key';
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('quota exhausted', { status: 429 })),
    );

    await expect(
      new KiConnectProvider().complete({
        system: 'S',
        messages: [{ role: 'user', content: 'x' }],
      }),
    ).rejects.toMatchObject({ status: 429 });
  });

  it('maps a network failure to 502', async () => {
    process.env.KICONNECT_API_KEY = 'kc-key';
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('fetch failed');
      }),
    );

    await expect(
      new KiConnectProvider().complete({
        system: 'S',
        messages: [{ role: 'user', content: 'x' }],
      }),
    ).rejects.toMatchObject({ status: 502 });
  });

  it('throws 502 when the body carries an error field', async () => {
    process.env.KICONNECT_API_KEY = 'kc-key';
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ error: { message: 'bad model' } }), { status: 200 }),
      ),
    );

    await expect(
      new KiConnectProvider().complete({
        system: 'S',
        messages: [{ role: 'user', content: 'x' }],
      }),
    ).rejects.toMatchObject({ status: 502 });
  });

  it('throws on an empty successful response', async () => {
    process.env.KICONNECT_API_KEY = 'kc-key';
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ choices: [] }), { status: 200 }),
      ),
    );

    await expect(
      new KiConnectProvider().complete({
        system: 'S',
        messages: [{ role: 'user', content: 'x' }],
      }),
    ).rejects.toBeInstanceOf(LlmError);
  });

  it('streams SSE deltas and yields text chunks', async () => {
    process.env.KICONNECT_API_KEY = 'kc-key';

    // Build a ReadableStream from raw SSE bytes.
    const sse = [
      'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
      'data: {"choices":[{"delta":{"content":" world"}}]}\n',
      'data: [DONE]\n',
    ].join('');
    const encoder = new TextEncoder();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(encoder.encode(sse), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        }),
      ),
    );

    const provider = new KiConnectProvider();
    const chunks: string[] = [];
    for await (const chunk of provider.stream({
      system: 'S',
      messages: [{ role: 'user', content: 'hi' }],
    })) {
      chunks.push(chunk);
    }
    expect(chunks.join('')).toBe('Hello world');
  });

  it('throws 503 from stream() when the key is missing', async () => {
    const provider = new KiConnectProvider();
    const gen = provider.stream({ system: 'S', messages: [{ role: 'user', content: 'x' }] });
    await expect(gen[Symbol.asyncIterator]().next()).rejects.toMatchObject({ status: 503 });
  });
});

describe('extractKiConnectDelta', () => {
  it('extracts the content delta from a data line', () => {
    const line = 'data: {"choices":[{"delta":{"content":"Hello"}}]}';
    expect(extractKiConnectDelta(line)).toBe('Hello');
  });

  it('returns null for the [DONE] sentinel', () => {
    expect(extractKiConnectDelta('data: [DONE]')).toBeNull();
  });

  it('returns null for non-data lines and keep-alives', () => {
    expect(extractKiConnectDelta(': keep-alive')).toBeNull();
    expect(extractKiConnectDelta('')).toBeNull();
    expect(extractKiConnectDelta('event: ping')).toBeNull();
  });

  it('returns null when the delta has no content (e.g. role-only chunk)', () => {
    expect(
      extractKiConnectDelta('data: {"choices":[{"delta":{"role":"assistant"}}]}'),
    ).toBeNull();
  });

  it('returns null on malformed JSON instead of throwing', () => {
    expect(extractKiConnectDelta('data: {not json')).toBeNull();
  });

  it('returns null for an empty data payload', () => {
    expect(extractKiConnectDelta('data: ')).toBeNull();
  });
});
