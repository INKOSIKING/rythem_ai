import React, { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import "./styles/global.css";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-50">
      <header className="p-4 text-3xl font-bold">Rhythm AI</header>
      <main className="max-w-2xl mx-auto mt-8">
        <ChatPanel />
      </main>
    </div>
  );
}