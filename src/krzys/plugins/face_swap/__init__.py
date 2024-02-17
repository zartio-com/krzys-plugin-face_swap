import krzys.core
from krzys.plugins.face_swap.cog import Cog


class Plugin(krzys.core.Plugin):
    async def load(self):
        self.init_face_swap()
        await self.bot.add_cog(Cog(self.bot))

    def init_face_swap(self):
        from .face_swap import face_analyzer
