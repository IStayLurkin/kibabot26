# Future Ideas

## Models

### Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled
- Fine-tuned Qwen3.5-27B distilled on Claude 4.6 Opus reasoning chains
- Native "developer" role support (works with Claude Code / OpenCode out of the box)
- Thinking mode fully preserved, ~16.5GB VRAM at Q4_K_M, 29-35 tok/s on 3090
- Stable tool-calling, can run autonomously for 9+ minutes without stalling
- **Use case:** swap in as the backend for `!think fast/best` or `!code ask fast/best`
- Not good for Koba chat — `<think>` blocks and agent-style reasoning would be jarring
- HuggingFace: `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`
