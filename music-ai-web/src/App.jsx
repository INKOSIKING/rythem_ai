import React, { useState } from "react";
import { MusicAI } from "music-ai-sdk";

const ai = new MusicAI("YOUR_API_KEY", "http://localhost:8000");

export default function App() {
  const [msg, setMsg] = useState("");
  const [history, setHistory] = useState([]);
  const [audioUrl, setAudioUrl] = useState(null);

  async function send() {
    const res = await ai.chat(msg);
    setHistory(history => [...history, { user: "me", text: msg }, { user: "AI", text: res.response }]);
    setMsg("");
    // Optionally trigger music generation here
  }

  return (
    <div style={{maxWidth: 600, margin: "2em auto", fontFamily: "sans-serif"}}>
      <h1>Music AI Chat</h1>
      <div style={{minHeight:200, border:"1px solid #ccc", padding:10, marginBottom:10}}>
        {history.map((h, i) => <div key={i}><b>{h.user}:</b> {h.text}</div>)}
      </div>
      <input value={msg} onChange={e => setMsg(e.target.value)} style={{width:"80%"}} />
      <button onClick={send}>Send</button>
      {audioUrl && <audio src={audioUrl} controls style={{marginTop:10}} />}
    </div>
  );
}