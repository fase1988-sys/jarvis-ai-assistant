import { LLMProvider, LLMResponse, StreamCallbacks, ToolDefinition } from '../llm-provider';

export class NvidiaNimProvider implements LLMProvider {
  readonly name = 'NVIDIA NIM';
  private readonly baseUrl = 'https://integrate.api.nvidia.com/v1';

  constructor(
    private readonly apiKey: string,
    private readonly model: string = 'meta/llama-3.1-70b-instruct'
  ) {}

  async generateText(prompt: string, systemPrompt?: string, callbacks?: StreamCallbacks): Promise<string> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        messages: [
          ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
          { role: 'user', content: prompt }
        ],
        stream: false
      })
    });
    if (!response.ok) throw new Error(`NVIDIA NIM API error: ${response.status} ${await response.text()}`);
    const data = await response.json() as any;
    const text = data.choices?.[0]?.message?.content ?? '';
    callbacks?.onToken?.(text);
    callbacks?.onComplete?.(text);
    return text;
  }

  async callWithTools(prompt: string, tools: ToolDefinition[], systemPrompt?: string): Promise<LLMResponse> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        messages: [
          ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
          { role: 'user', content: prompt }
        ],
        tools: tools.map(t => ({ type: 'function', function: t })),
        tool_choice: 'auto',
        stream: false
      })
    });
    if (!response.ok) throw new Error(`NVIDIA NIM API error: ${response.status} ${await response.text()}`);
    const data = await response.json() as any;
    const message = data.choices?.[0]?.message ?? {};
    const call = message.tool_calls?.[0];
    if (call?.function) {
      let args: Record<string, any> = {};
      try { args = JSON.parse(call.function.arguments || '{}'); } catch { args = {}; }
      return { type: 'tool_call', toolCall: { name: call.function.name, arguments: args } };
    }
    return { type: 'text', text: message.content ?? '' };
  }

  supportsToolCalling(): boolean { return true; }
}
