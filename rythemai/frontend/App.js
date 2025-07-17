import React, { useState } from "react";
import axios from "axios";

export default function App() {
  const [chat, setChat] = useState("");
  const [history, setHistory] = useState([]);
  const [audio, setAudio] = useState(null);

  async function sendChat() {
    const res = await axios.post("/api/chat/", { user_id: "u1", message: chat });
    setHistory([...history, { user: "me", text: chat }, { user: "AI", text: res.data.response }]);
    if (res.data.music_id) {
      const musicRes = await axios.get(`/api/music/${res.data.music_id}`, { responseType: "arraybuffer" });
      setAudio(URL.createObjectURL(new Blob([musicRes.data])));
    }
    setChat("");
  }

  return (
    <div>
      <div>{history.map((m, i) => <div key={i}><b>{m.user}:</b> {m.text}</div>)}</div>
      <input value={chat} onChange={e => setChat(e.target.value)} />
      <button onClick={sendChat}>Send</button>
      {audio && <audio src={audio} controls />}
    </div>
  );
}