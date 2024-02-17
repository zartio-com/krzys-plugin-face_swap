import asyncio
import io
from queue import Queue
from urllib.request import urlopen

import discord
from discord.ext import commands
from discord import app_commands

from krzys.plugins.face_swap.exception import SourceFileDoesNotContainFaceError, TargetImageDoesNotContainFaceError
from krzys.plugins.face_swap.face_swap import process_image, is_image, process_video_multithreaded2, get_one_face


class QueueItem:
    def __init__(self, interaction: discord.Interaction, source: discord.Attachment):
        self.interaction = interaction
        self._source = source

    @property
    def source_is_image(self) -> bool:
        return is_image(self._source.url)

    async def download_source(self) -> bytes:
        return await self._source.read()

    @property
    def target_is_image(self) -> bool:
        return True

    async def download_target(self) -> bytes | None:
        return None


class TargetAttachmentQueueItem(QueueItem):
    def __init__(self, interaction: discord.Interaction, source: discord.Attachment, target: discord.Attachment):
        super().__init__(interaction, source)
        self._target = target

    @property
    def target_is_image(self) -> bool:
        return is_image(self._target.url)

    async def download_target(self) -> bytes | None:
        # noinspection PyBroadException
        try:
            return await self._target.read()
        except Exception:
            return None


class TargetUrlQueueItem(QueueItem):
    def __init__(self, interaction: discord.Interaction, source: discord.Attachment, target: str):
        super().__init__(interaction, source)
        self._target = target

    def target_is_image(self) -> bool:
        return is_image(self._target)

    async def download_target(self) -> bytes | None:
        # noinspection PyBroadException
        try:
            return urlopen(self._target).read()
        except Exception:
            return None


class Cog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot: commands.Bot = bot
        self.is_busy = False
        self.queue = Queue()
        self.bot.loop.create_task(self.queue_worker())

    @app_commands.command(name='faceswap')
    async def face_swap(self, i: discord.Interaction, source_face: discord.Attachment, target: discord.Attachment):
        r: discord.InteractionResponse = i.response
        await r.defer()

        self.queue.put(TargetAttachmentQueueItem(i, source_face, target))

    @app_commands.command(name='faceswap_url')
    async def face_swap_url(self, i: discord.Interaction, source_face: discord.Attachment, target: str):
        r: discord.InteractionResponse = i.response
        await r.defer()

        self.queue.put(TargetUrlQueueItem(i, source_face, target))

    async def queue_worker(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(1)
                continue

            item: QueueItem = self.queue.get()

            if not item.source_is_image:
                await item.interaction.edit_original_response(content='Źródło musi być obrazkiem')
                continue

            await item.interaction.edit_original_response(content="Pobieram załączone treści...")
            source = await item.download_source()
            target = await item.download_target()

            if source is None or target is None:
                await item.interaction.edit_original_response(content='Nie udało się pobrać załączonych treści')
                continue

            source_face = get_one_face(source)
            if source_face is None:
                await item.interaction.edit_original_response(content='Nie znaleziono twarzy w źródle')
                continue

            # noinspection PyBroadException
            try:
                if item.target_is_image:
                    result = process_image(source_face=source_face, target=target)

                    await item.interaction.edit_original_response(content='', attachments=[
                        discord.File(io.BytesIO(result), filename='result.jpg')])
                    continue

                result = await process_video_multithreaded2(
                    i=item.interaction, source=source, target=target, threads=24, fps=18)
                await item.interaction.edit_original_response(content='', attachments=[
                    discord.File(io.BytesIO(result), filename='result.mp4')])
            except TargetImageDoesNotContainFaceError:
                await item.interaction.edit_original_response(content='Nie znaleziono twarzy w obrazku docelowym')
            except Exception:
                await item.interaction.edit_original_response(content='FFmpeg spadł z rowerka')
