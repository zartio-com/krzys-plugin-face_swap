import io

import discord
from discord.ext import commands
from discord import app_commands

from krzys.plugins.face_swap.exception import SourceFileDoesNotContainFaceError, TargetImageDoesNotContainFaceError
from krzys.plugins.face_swap.face_swap import process_image, is_image, process_video_multithreaded2


class Cog(commands.Cog):
    image_extensions = ['png', 'jpg', 'jpeg']

    def __init__(self, bot: commands.Bot):
        self.bot: commands.Bot = bot
        self.is_busy = False

    @app_commands.command(name='faceswap')
    async def face_swap(self, i: discord.Interaction, source_face: discord.Attachment, target: discord.Attachment):
        r: discord.InteractionResponse = i.response
        await r.defer()

        if self.is_busy:
            await i.followup.send('Wypierdalać, jestem zajęty!')
            return

        if not is_image(source_face.url):
            await i.followup.send('Źródło musi być obrazkiem')
            return

        result: bytes | None = None
        if is_image(target.url):
            self.is_busy = True
            try:
                result = process_image(source=await source_face.read(), target=await target.read())
            except SourceFileDoesNotContainFaceError:
                await i.followup.send('Nie znaleziono twarzy w źródle')
            except TargetImageDoesNotContainFaceError:
                await i.followup.send('Nie znaleziono twarzy w obrazku docelowym')
            except Exception:
                await i.followup.send('Niezintendytyfikowany błąd')

            self.is_busy = False
            if result is not None:
                await i.followup.send(file=discord.File(io.BytesIO(result), filename='result.png'))

            return

        self.is_busy = True
        try:
            result = await process_video_multithreaded2(
                i=i, source=await source_face.read(), target=await target.read(), threads=24, fps=18)
        except SourceFileDoesNotContainFaceError:
            await i.followup.send('Nie znaleziono twarzy w źródle')
        except Exception:
            await i.followup.send('Niezintendytyfikowany błąd')

        self.is_busy = False
        if result is not None:
            await i.followup.send(file=discord.File(io.BytesIO(result), filename='result.mp4'))
