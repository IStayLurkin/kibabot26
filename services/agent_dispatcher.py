import os
import re
from typing import TypedDict, Annotated, List, Union, Optional
from langgraph.graph import StateGraph, END
from services.llm_service import LLMService
from services.image_service import ImageService
from services.music_service import MusicService
from database.chat_memory import (
    get_user_memory, 
    get_conversation_summary, 
    get_conversation_state
)

# Define the Agent State
class AgentState(TypedDict):
    messages: List[str]
    user_id: str
    channel_id: str
    next_step: str
    file_path: Optional[str]
    display_name: str

class AgentDispatcher:
    def __init__(self, bot):
        self.bot = bot
        self.llm = getattr(bot, "llm_service", None) or LLMService()
        self.image_gen = getattr(bot, "image_service", None) or ImageService()
        self.music_gen = getattr(bot, "music_service", None) or MusicService()
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Builds the LangGraph routing logic for Kiba's 3090 Ti Multimodal Hub."""
        graph = StateGraph(AgentState)

        # Add Nodes
        graph.add_node("router", self.router_node)
        graph.add_node("chat_agent", self.coding_node)
        graph.add_node("media_agent", self.media_node)
        graph.add_node("music_agent", self.music_agent_node)

        # Define Flow
        graph.set_entry_point("router")

        # Mapping logic based on detected intent
        graph.add_conditional_edges(
            "router",
            lambda x: x["next_step"],
            {
                "chat": "chat_agent",
                "draw": "media_agent",
                "sing": "music_agent"
            }
        )

        graph.add_edge("chat_agent", END)
        graph.add_edge("media_agent", END)
        graph.add_edge("music_agent", END)

        return graph.compile()

    def classify_intent(self, content: str) -> str:
        lowered = content.lower()
        words = set(re.findall(r'\b\w+\b', lowered))
        
        creative_override = any(word in words for word in ["imagine", "fictional", "pretend"]) or \
                            any(phrase in lowered for phrase in ["make up", "create a picture"])
        
        music_triggers = {"sing", "song", "melody", "music", "audio", "vocal", "lyrics"}
        media_triggers = {"draw", "generate", "image", "visual", "flux"}

        if music_triggers & words:
            if not creative_override and "real" in words:
                return "chat"
            return "sing"

        if media_triggers & words or "create a picture" in lowered:
            if not creative_override and "real" in words:
                return "chat"
            return "draw"

        return "chat"

    async def router_node(self, state: AgentState):
        next_step = self.classify_intent(state["messages"][-1])
        return {**state, "next_step": next_step}

    async def coding_node(self, state: AgentState):
        """Dialed-in node that injects Brandon's memory and hardware context."""
        user_id = state["user_id"]
        channel_id = state.get("channel_id", "default")
        display_name = state.get("display_name") or user_id
        prompt = state["messages"][-1]

        # 1. Retrieve Brandon's facts and recent history from the G: drive DB
        memory_rows = await get_user_memory(user_id)
        user_context = "\n".join([f"- {k}: {v}" for k, v in memory_rows])
        summary = await get_conversation_summary(user_id, channel_id)
        state_data = await get_conversation_state(user_id, channel_id)

        # 2. Build the hardware-aware preamble
        # This forces the model to acknowledge its 3090 Ti environment
        hardware_context = (
            "SYSTEM CONTEXT:\n"
            "Environment: Local RTX 3090 Ti | 24GB VRAM | G: Drive Storage.\n"
            f"User Identity: {user_context if user_context else 'Owner'}.\n"
            f"Recent Summary: {summary}\n"
        )

        # 3. Call LLM with full context injection
        response = await self.llm.generate_reply(
            user_display_name=display_name,
            user_message=f"{hardware_context}\n\nUser Request: {prompt}",
            memory=dict(memory_rows),
            recent_messages=[], # LLMService handles history via session_id
            conversation_summary=summary,
            intent_category=state_data.get("last_intent"),
            conversation_goal=state_data.get("goal")
        )

        return {"messages": [response]}

    async def media_node(self, state: AgentState):
        """FLUX.2 Image generation path with Safety Lock for VRAM Guard."""
        async with self.bot.generating_lock:
            self.bot.generating_count += 1
            try:
                prompt = state["messages"][-1]
                image_path = await self.image_gen.generate_image(prompt)

                if image_path and os.path.exists(image_path):
                    return {
                        "messages": ["🎨 **Kiba Engine:** Image rendering complete. Patching file..."],
                        "file_path": image_path,
                    }
                return {"messages": ["❌ Image generation failed. Check terminal logs."], "file_path": None}
            finally:
                self.bot.generating_count -= 1

    async def music_agent_node(self, state: AgentState):
        """YuE / Stable Audio generation path with Safety Lock for VRAM Guard."""
        async with self.bot.generating_lock:
            self.bot.generating_count += 1
            try:
                prompt = state["messages"][-1]

                if any(word in prompt.lower() for word in ["sing", "lyrics", "song", "vocal"]):
                    audio_path = await self.music_gen.generate_song_clip(
                        vibe="cinematic",
                        bpm=120,
                        voice_style="studio",
                        vocal_mode="lyrics",
                        lyrics=prompt,
                    )
                else:
                    audio_path = await self.music_gen.generate_melody(prompt)

                if audio_path and os.path.exists(audio_path):
                    return {
                        "messages": ["🎵 **Studio Specialist:** Audio synthesis complete. Uploading track..."],
                        "file_path": audio_path,
                    }
                return {"messages": ["❌ Audio generation failed. Check VRAM availability."], "file_path": None}
            finally:
                self.bot.generating_count -= 1

    async def run(self, user_id: str, channel_id: str, content: str, display_name: str = ""):
        """Dispatcher entry point. Now correctly routes channel_id for memory lookup."""
        inputs = {
            "messages": [content],
            "user_id": user_id,
            "channel_id": channel_id,
            "next_step": "",
            "file_path": None,
            "display_name": display_name or user_id,
        }
        result = await self.workflow.ainvoke(inputs)
        return result["messages"][-1], result.get("file_path")