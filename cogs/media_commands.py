import os
from __future__ import annotations
import os, json, time

from pathlib import Path

import discord
from discord.ext import commands

from core.config import MAX_PROMPT_LENGTH, MAX_TTS_LENGTH
from core.feature_flags import IMAGE_ENABLED, VIDEO_ENABLED, VOICE_ENABLED
from services.image_service import ImageService
from services.music_service import MusicService
from services.video_service import VideoService
from services.voice_service import VoiceService


class MediaCommands(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        llm_service = getattr(bot, "llm_service", None)
        performance_tracker = getattr(bot, "performance_tracker", None)
        self.image_service = getattr(
            bot,
            "image_service",
            ImageService(llm_service=llm_service, performance_tracker=performance_tracker),
        )
        self.voice_service = getattr(
            bot,
            "voice_service",
            VoiceService(llm_service=llm_service, performance_tracker=performance_tracker),
        )
        self.video_service = getattr(
            bot,
            "video_service",
            VideoService(llm_service=llm_service, performance_tracker=performance_tracker),
        )
        self.music_service = getattr(
            bot,
            "music_service",
            MusicService(performance_tracker=performance_tracker),
        )

    @commands.command(name="image", aliases=["img"])
    async def image_command(self, ctx: commands.Context, *, prompt: str) -> None:
        if not IMAGE_ENABLED:
            await ctx.send("Image generation is disabled.")
            return

        prompt = prompt.strip()
        if not prompt:
            await ctx.send("Provide a prompt. Example: `!image a neon fox in the rain`")
            return

        if len(prompt) > MAX_PROMPT_LENGTH:
            await ctx.send(f"Prompt is too long. Keep it under {MAX_PROMPT_LENGTH} characters.")
            return

        await ctx.send("On it, generating that now...")
        async with ctx.typing():
            try:
                image_path = await self.image_service.generate_image(prompt)
                await ctx.send(file=discord.File(image_path, filename=Path(image_path).name))
            except Exception as exc:
                await ctx.send(f"Image generation failed: {exc}")

    @commands.command(name="tts", aliases=["say"])
    async def tts_command(self, ctx: commands.Context, *, text: str) -> None:
        if not VOICE_ENABLED:
            await ctx.send("Voice generation is disabled.")
            return

        text = text.strip()
        if not text:
            await ctx.send("Provide text to speak. Example: `!tts hello from Kiba Bot`")
            return

        if len(text) > MAX_TTS_LENGTH:
            await ctx.send(f"Text is too long. Keep it under {MAX_TTS_LENGTH} characters.")
            return

        await ctx.send("On it, making that audio now...")
        async with ctx.typing():
            try:
                audio_path = await self.voice_service.text_to_speech(text)
                await ctx.send(file=discord.File(audio_path, filename=Path(audio_path).name))
            except Exception as exc:
                await ctx.send(f"Text-to-speech failed: {exc}")

    @commands.command(name="video", aliases=["animate"])
    async def video_command(self, ctx: commands.Context, *, prompt: str) -> None:
        if not VIDEO_ENABLED:
            await ctx.send("Video generation is disabled.")
            return

        prompt = prompt.strip()
        if not prompt:
            await ctx.send("Provide a prompt. Example: `!video a flying dragon over mountains`")
            return

        if len(prompt) > MAX_PROMPT_LENGTH:
            await ctx.send(f"Prompt is too long. Keep it under {MAX_PROMPT_LENGTH} characters.")
            return

        await ctx.send("On it, starting that video request now...")
        async with ctx.typing():
            try:
                video_path = await self.video_service.generate_video(prompt)
                await ctx.send(file=discord.File(video_path, filename=Path(video_path).name))
            except NotImplementedError as exc:
                await ctx.send(str(exc))
            except Exception as exc:
                await ctx.send(f"Video generation failed: {exc}")

    @commands.command(name="melody", aliases=["music", "tune"])
    async def melody_command(self, ctx: commands.Context, *, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            await ctx.send("Provide a prompt. Example: `!melody calm dreamy piano loop`")
            return

        if len(prompt) > MAX_PROMPT_LENGTH:
            await ctx.send(f"Prompt is too long. Keep it under {MAX_PROMPT_LENGTH} characters.")
            return

        await ctx.send("On it, composing that melody now...")
        async with ctx.typing():
            try:
                melody_path = await self.music_service.generate_melody(prompt)
                await ctx.send(file=discord.File(melody_path, filename=Path(melody_path).name))
            except Exception as exc:
                await ctx.send(f"Melody generation failed: {exc}")

    @commands.command(name="song", aliases=["vocals", "sing"])
    async def song_command(self, ctx: commands.Context) -> None:
        song_session_service = getattr(self.bot, "song_session_service", None)
        if song_session_service is None:
            await ctx.send("Song generation setup is not available right now.")
            return

        question = song_session_service.begin_session(str(ctx.author.id), str(ctx.channel.id))
        await ctx.send(
            "Let's build your vocal clip.\n"
            f"{question}"
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MediaCommands(bot))
