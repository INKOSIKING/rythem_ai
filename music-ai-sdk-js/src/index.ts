export class MusicAI {
  constructor(private apiKey: string, private baseUrl = "https://api.yourmusicai.com") {}

  async chat(message: string, userId = "user1") {
    const res = await fetch(`${this.baseUrl}/chat/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({ user_id: userId, message }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async generateMusic(seed: number[]) {
    const res = await fetch(`${this.baseUrl}/generate-music/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": `Bearer ${this.apiKey}`
      },
      body: new URLSearchParams({ seed: JSON.stringify(seed) }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
}