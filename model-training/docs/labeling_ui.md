# Rhythm AI – Dataset Labeling & Composer UI

## Features

- **Audio/MIDI/lyric annotation in browser**
- **Tagging, genre detection, beat alignment**
- **Web-based upload for new data**
- **Support for collaborative multi-user workflows**
- **Admin review/approval for production datasets**

## Recommended Tools

- [Label Studio](https://labelstud.io/) for custom audio, text, and sequence annotation.
- [Prodigy](https://prodi.gy/) for advanced tagging.
- [Heartex](https://heartex.com/) for enterprise-scale annotation.

## Minimal React/Next.js UI Example

```jsx
import React, { useRef, useState } from 'react';

export default function AudioLabeler() {
  const audioRef = useRef();
  const [labels, setLabels] = useState([]);
  const handleLabel = (tag) => setLabels([...labels, tag]);

  return (
    <div>
      <audio ref={audioRef} src="/your-audio.wav" controls />
      <button onClick={() => handleLabel("Jazz")}>Jazz</button>
      <button onClick={() => handleLabel("HipHop")}>HipHop</button>
      <button onClick={() => handleLabel("Energetic")}>Energetic</button>
      <div>Labels: {labels.join(", ")}</div>
    </div>
  );
}
```

## Integration

- Export labels as JSON for direct training integration.
- Connect to cloud storage for uploads and downloads.
- Add login/auth for collaborative workflows.

---

**For full-featured enterprise labeling, deploy [Label Studio](https://labelstud.io/) on your cloud and connect to your dataset buckets.**