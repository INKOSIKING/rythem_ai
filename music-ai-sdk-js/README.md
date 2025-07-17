# music-ai-sdk-js

Enterprise-grade JS/TS SDK for Music AI backend.

- Robust error handling
- Easy integration for web/mobile/desktop

## Usage

```js
import { MusicAI } from "music-ai-sdk";
const ai = new MusicAI("YOUR_API_KEY");
const resp = await ai.chat("Make a funky bassline.");
const music = await ai.generateMusic([1,2,3,4,5]);
```