import React, { useState } from "react";
import { MusicAI } from "rhythm-ai-sdk";
import "tailwindcss/tailwind.css";

const api = new MusicAI(process.env.REACT_APP_API_KEY);

export default function App() {
  const [msg, setMsg] = useState("");
  const [history, setHistory] = useState([]);
  const [audio, setAudio] = useState(null);

  async function send() {
    const resp = await api.chat(msg);
    setHistory([...history, { user: "me", text: msg }, { user: "AI", text: resp.result }]);
    setMsg("");
    // Optionally, call audio generation here.
  }

  return (
    <div className="max-w-2xl m-auto p-8">
      <h1 className="text-3xl font-bold mb-4">Rhythm AI</h1>
      <div className="mb-4">
        {history.map((h, i) => <div key={i}><b>{h.user}:</b> {h.text}</div>)}
      </div>
      <input className="border p-2 w-4/5" value={msg} onChange={e => setMsg(e.target.value)} />
      <button className="bg-blue-500 text-white p-2 ml-2" onClick={send}>Send</button>
      {audio && <audio src={audio} controls className="mt-4" />}
    </div>
  );
}