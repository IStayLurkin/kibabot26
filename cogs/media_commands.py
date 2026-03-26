from __future__ import annotations

from pathlib import Path

import discord
from discord.ext import commands

from core.config import MAX_PROMPT_LENGTH, MAX_TTS_LENGTH
from core.feature_flags import IMAGE_ENABLED, VIDEO_ENABLED, VOICE_ENABLED
from services.image_service import ImageService
from services.music_service import MusicService
from services.video_service import VideoService
from services.voice_service import VoiceService


_DISCORD_FILE_LIMIT_BYTES = 25 * 1024 * 1024  # 25 MB — standard server limit


def _check_file_size(path: str) -> bool:
    """Returns True if file is within Discord's upload limit."""
    try:
        return Path(path).stat().st_size <= _DISCORD_FILE_LIMIT_BYTES
    except OSError:
        return False


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
            MusicService(
                performance_tracker=performance_tracker,
                runtime_service=getattr(bot, "model_runtime_service", None),
            ),
        )

    @commands.command(name="image", aliases=["img"])
    @commands.cooldown(1, 30, commands.BucketType.user)
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
                if image_path and Path(image_path).exists():
                    if not _check_file_size(image_path):
                        await ctx.send("❌ Generated image is too large to upload (>25MB).")
                    else:
                        await ctx.send(file=discord.File(image_path, filename=Path(image_path).name))
                else:
                    await ctx.send("❌ Image generation failed. Check VRAM availability.")
            except Exception as exc:
                await ctx.send(f"Image generation failed: {exc}")

    @commands.group(name="tts", aliases=["say"], invoke_without_command=True)
    async def tts_group(self, ctx: commands.Context, *, text: str = "") -> None:
        if not text:
            await ctx.send("Usage: `!tts fast <text>` (Piper) or `!tts best <text>` (Fish Speech)\nOr just `!tts <text>` to use fast tier.")
            return
        await self._run_tts(ctx, text, "fast")

    @tts_group.command(name="fast")
    async def tts_fast(self, ctx: commands.Context, *, text: str) -> None:
        await self._run_tts(ctx, text, "fast")

    @tts_group.command(name="best")
    async def tts_best(self, ctx: commands.Context, *, text: str) -> None:
        await self._run_tts(ctx, text, "best")

    async def _run_tts(self, ctx: commands.Context, text: str, tier: str) -> None:
        if not VOICE_ENABLED:
            await ctx.send("Voice generation is disabled.")
            return
        text = text.strip()
        if not text:
            await ctx.send("Provide text to speak.")
            return
        if len(text) > MAX_TTS_LENGTH:
            await ctx.send(f"Text is too long. Keep it under {MAX_TTS_LENGTH} characters.")
            return
        async with ctx.typing():
            try:
                if tier == "best":
                    fish = getattr(self.bot, "fish_speech_service", None)
                    if fish is None:
                        await ctx.send("Fish Speech is not enabled. Set `FISH_SPEECH_ENABLED=true` in .env.")
                        return
                    audio_path = await fish.synthesize(text)
                else:
                    audio_path = await self.voice_service.text_to_speech(text)
                if audio_path and Path(audio_path).exists():
                    if not _check_file_size(audio_path):
                        await ctx.send("❌ TTS audio is too large to upload (>25MB).")
                    else:
                        await ctx.send(file=discord.File(audio_path, filename=Path(audio_path).name))
                else:
                    await ctx.send("❌ TTS failed.")
            except Exception as exc:
                await ctx.send(f"Text-to-speech failed: {exc}")

    @commands.group(name="stt", invoke_without_command=True)
    async def stt_group(self, ctx: commands.Context) -> None:
        from database.behavior_rules_repository import get_bot_config
        tier = await get_bot_config("stt_tier", "fast")
        await ctx.send(f"Current STT tier: `{tier}` (fast=Whisper, best=Parakeet)\nUse `!stt fast` or `!stt best` to switch.")

    @stt_group.command(name="fast")
    async def stt_fast(self, ctx: commands.Context) -> None:
        from database.behavior_rules_repository import set_bot_config
        await set_bot_config("stt_tier", "fast")
        await ctx.send("STT set to `fast` (Faster-Whisper).")

    @stt_group.command(name="best")
    async def stt_best(self, ctx: commands.Context) -> None:
        from database.behavior_rules_repository import set_bot_config
        parakeet = getattr(self.bot, "parakeet_service", None)
        if parakeet is None:
            await ctx.send("Parakeet is not enabled. Set `PARAKEET_ENABLED=true` in .env and restart.")
            return
        await set_bot_config("stt_tier", "best")
        await ctx.send("STT set to `best` (Parakeet V3).")

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
                if video_path and Path(video_path).exists():
                    if not _check_file_size(video_path):
                        await ctx.send("❌ Generated video is too large to upload (>25MB). Check outputs folder.")
                    else:
                        await ctx.send(file=discord.File(video_path, filename=Path(video_path).name))
                else:
                    await ctx.send("❌ Video generation failed. Check VRAM availability.")
            except NotImplementedError as exc:
                await ctx.send(str(exc))
            except Exception as exc:
                await ctx.send(f"Video generation failed: {exc}")

    @commands.command(name="melody", aliases=["music", "tune"])
    async def melody_command(self, ctx: commands.Context, *, prompt: str) -> None:
        if self.music_service is None:
            await ctx.send("❌ Music service is not loaded.")
            return

        prompt = prompt.strip()
        if not prompt:
            await ctx.send("Provide a prompt. Example: `!melody calm dreamy piano loop`")
            return

        if len(prompt) > MAX_PROMPT_LENGTH:
            await ctx.send(f"Prompt is too long. Keep it under {MAX_PROMPT_LENGTH} characters.")
            return

        await ctx.send("🎵 On it, composing that melody now... (this takes a minute)")
        async with ctx.typing():
            try:
                melody_path = await self.music_service.generate_melody(prompt)
                if melody_path and Path(melody_path).exists():
                    if not _check_file_size(melody_path):
                        await ctx.send("❌ Generated audio is too large to upload (>25MB).")
                    else:
                        await ctx.send(file=discord.File(melody_path, filename=Path(melody_path).name))
                else:
                    await ctx.send("❌ Melody generation failed. Check VRAM availability.")
            except Exception as exc:
                await ctx.send(f"❌ Melody generation failed: {exc}")

    @commands.command(name="song", aliases=["vocals", "sing"])
    async def song_command(self, ctx: commands.Context, *, prompt: str) -> None:
        """Generate a full vocal song clip via YuE. Usage: !song <genre/vibe>. <lyrics>"""
        if self.music_service is None:
            await ctx.send("❌ Music service is not loaded.")
            return

        prompt = prompt.strip()
        if not prompt:
            await ctx.send(
                "Provide a vibe and optional lyrics.\n"
                "Example: `!song cinematic epic female vocal. We rise at dawn, we fight till dusk`"
            )
            return

        await ctx.send("🎤 Starting YuE song generation... (this takes ~6 minutes on 3090 Ti)")
        async with ctx.typing():
            try:
                audio_path = await self.music_service.generate_song_clip(
                    vibe=prompt.split(".")[0].strip(),
                    bpm=self.music_service.bpm,
                    voice_style=self.music_service.voice_style,
                    vocal_mode=self.music_service.vocal_mode,
                    lyrics=prompt.split(".", 1)[1].strip() if "." in prompt else prompt,
                )
                if audio_path and Path(audio_path).exists():
                    if not _check_file_size(audio_path):
                        await ctx.send("❌ Generated song is too large to upload (>25MB). Check outputs folder.")
                    else:
                        await ctx.send(file=discord.File(audio_path, filename=Path(audio_path).name))
                else:
                    await ctx.send("❌ Song generation failed. Check VRAM availability and YuE repo path.")
            except Exception as exc:
                await ctx.send(f"❌ Song generation failed: {exc}")

    @image_command.error
    async def image_cooldown_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"⏳ Cooldown — try again in {error.retry_after:.0f}s.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MediaCommands(bot))
