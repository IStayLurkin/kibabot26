import os
from typing import TypedDict, Annotated, List, Union, Optional
from langgraph.graph import StateGraph, END
from services.llm_service import LLMService
from services.image_service import ImageService
from services.music_service import MusicService

# Define the Agent State
class AgentState(TypedDict):
    messages: List[str]
    user_id: str
    channel_id: str
    next_step: str
    file_path: Optional[str]

class AgentDispatcher:
    def __init__(self, bot):
        self.bot = bot
        self.llm = getattr(bot, "llm_service", LLMService())
        self.image_gen = getattr(bot, "image_service", ImageService())
        self.music_gen = getattr(bot, "music_service", MusicService())
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Builds the LangGraph routing logic for Kiba's 3090 Ti Multimodal Hub."""
        graph = StateGraph(AgentState)

        # Add Nodes
        graph.add_node("router", self.router_node)
        graph.add_node("coding_agent", self.coding_node)
        graph.add_node("media_agent", self.media_node)
        graph.add_node("music_agent", self.music_agent_node)

        # Define Flow
        graph.set_entry_point("router")
        
        # Mapping logic based on detected intent
        graph.add_conditional_edges(
            "router",
            lambda x: x["next_step"],
            {
                "chat": "coding_agent",
                "draw": "media_agent",
                "sing": "music_agent"
            }
        )

        graph.add_edge("coding_agent", END)
        graph.add_edge("media_agent", END)
        graph.add_edge("music_agent", END)

        return graph.compile()

    def router_node(self, state: AgentState):
        """Hardware-aware intent detection to prevent VRAM 'crossing wires'."""
        content = state["messages"][-1].lower()
        
        # Audio/Music Triggers
        music_triggers = ["sing", "song", "melody", "music", "audio", "vocal", "lyrics"]
        # Image Triggers
        media_triggers = ["draw", "generate", "image", "visual", "create a picture", "flux"]
        
        if any(trigger in content for trigger in music_triggers):
            return {**state, "next_step": "sing"}
        if any(trigger in content for trigger in media_triggers):
            return {**state, "next_step": "draw"}
            
        return {**state, "next_step": "chat"}

    async def coding_node(self, state: AgentState):
        """Standard LLM interaction via Qwen3-Coder (Resident Ollama)."""
        prompt = state["messages"][-1]
        response = await self.llm.generate_response(prompt)
        return {"messages": [response]}

    async def media_node(self, state: AgentState):
        """FLUX.2 Image generation path with Safety Lock for VRAM Guard."""
        self.bot.is_generating = True
        try:
            prompt = state["messages"][-1]
            # This handles the Ollama unload and FLUX.2 load internally
            image_path = await self.image_gen.generate_image(prompt)
            
            if image_path and os.path.exists(image_path):
                return {
                    "messages": ["🎨 **Kiba Engine:** Image rendering complete. Patching file..."],
                    "file_path": image_path 
                }
            return {"messages": ["❌ Image generation failed. Check terminal logs."], "file_path": None}
        finally:
            self.bot.is_generating = False

    async def music_agent_node(self, state: AgentState):
        """YuE / Stable Audio generation path with Safety Lock for VRAM Guard."""
        self.bot.is_generating = True
        try:
            prompt = state["messages"][-1]
            
            # Determine if it's a song or just a melody
            if any(word in prompt.lower() for word in ["sing", "lyrics", "song", "vocal"]):
                # Request a full song clip (YuE)
                audio_path = await self.music_gen.generate_song_clip(
                    vibe="cinematic", 
                    bpm=120, 
                    voice_style="studio", 
                    vocal_mode="lyrics",
                    lyrics=prompt
                )
            else:
                # Request a background melody (Stable Audio Open)
                audio_path = await self.music_gen.generate_melody(prompt)

            if audio_path and os.path.exists(audio_path):
                return {
                    "messages": ["🎵 **Studio Specialist:** Audio synthesis complete. Uploading track..."],
                    "file_path": audio_path
                }
            return {"messages": ["❌ Audio generation failed. Check VRAM availability."], "file_path": None}
        finally:
            self.bot.is_generating = False

    async def run(self, user_id: str, content: str):
        """Dispatcher entry point. Returns a tuple of (response_text, file_path_to_upload)."""
        inputs = {
            "messages": [content], 
            "user_id": user_id, 
            "next_step": "", 
            "file_path": None
        }
        result = await self.workflow.ainvoke(inputs)
        return result["messages"][-1], result.get("file_path")