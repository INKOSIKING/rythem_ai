import React, { useState } from "react";
import { View, TextInput, Button, Text, ScrollView } from "react-native";
import { MusicAI } from "music-ai-sdk";

const ai = new MusicAI("YOUR_API_KEY", "http://localhost:8000");

export default function App() {
  const [msg, setMsg] = useState('');
  const [history, setHistory] = useState([]);
  // Audio playback integration would use a suitable RN package

  async function send() {
    const res = await ai.chat(msg);
    setHistory(history => [...history, { user: "me", text: msg }, { user: "AI", text: res.response }]);
    setMsg('');
  }

  return (
    <View style={{padding:20}}>
      <ScrollView style={{height:300}}>
        {history.map((h, i) => <Text key={i}>{h.user}: {h.text}</Text>)}
      </ScrollView>
      <TextInput
        value={msg}
        onChangeText={setMsg}
        style={{borderWidth:1, borderColor:"#ccc", margin:10, padding:8}}
      />
      <Button title="Send" onPress={send} />
    </View>
  );
}