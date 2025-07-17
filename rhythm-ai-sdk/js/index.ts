export class MusicAI {
  constructor(private apiKey: string, private baseUrl = "https://api.rhythmai.com") {}
  async chat(message: string) {
    const res = await fetch(`${this.baseUrl}/chat/`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${this.apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: message }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
}