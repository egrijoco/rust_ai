import os
import asyncio
import discord
from discord import app_commands
from dotenv import load_dotenv
from rag import answer

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("DISCORD_GUILD_ID")  # opcionĂˇlis: gyorsabb sync egy szerverre

class RustRAGClient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

client = RustRAGClient()

@client.tree.command(name="rust", description="Rust kĂ©rdĂ©s magyarul (RAG + helyi LLM)")
@app_commands.describe(kerdes="Tedd fel a kĂ©rdĂ©sed")
async def rust(interaction: discord.Interaction, kerdes: str):
    await interaction.response.defer(thinking=True, ephemeral=False)
    try:
        # blokkolĂł hĂ­vĂˇs â†’ futtasd thread poolban
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, answer, kerdes)
        # Discord ĂĽzenethossz limit ~2000 karakter
        if len(out) > 1900:
            out = out[:1900] + "â€¦"
        await interaction.followup.send(out)
    except Exception as e:
        await interaction.followup.send(f"Hiba: {e}")

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("DISCORD_TOKEN hiĂˇnyzik a .env-bĹ‘l")
    client.run(TOKEN)
