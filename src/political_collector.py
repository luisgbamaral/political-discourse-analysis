import os
import asyncio
from datetime import datetime, timezone
import polars as pl
from telethon import TelegramClient
from dotenv import load_dotenv
from src.utils import get_logger, DATA_RAW

load_dotenv()
logger = get_logger("PoliticalCollector")

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION = 'research_session'

START_DATE = datetime(2018, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)

TARGETS = [
    ('jairbolsonarobrasil', 'Bolsonaro'),
    ('LulanoTelegram', 'Lula')
]

async def fetch_history(client: TelegramClient, channel: str, label: str) -> list[dict]:
    try:
        entity = await client.get_entity(channel)
    except ValueError:
        logger.error(f"Channel @{channel} not found.")
        return []

    data = []
    async for msg in client.iter_messages(entity, offset_date=END_DATE):
        if msg.date < START_DATE:
            break
        
        if msg.text:
            data.append({
                "date": msg.date.replace(tzinfo=None),
                "author": label,
                "text": msg.text
            })
            
    logger.info(f"@{channel}: {len(data)} messages retrieved.")
    return data

async def main():
    if not API_ID or not API_HASH:
        logger.critical("Missing credentials.")
        return

    async with TelegramClient(SESSION, int(API_ID), API_HASH) as client:
        logger.info("Starting collection...")
        
        tasks = [fetch_history(client, channel, label) for channel, label in TARGETS]
        results = await asyncio.gather(*tasks)
        
        raw_data = [item for sublist in results for item in sublist]

        if not raw_data:
            logger.warning("No data found.")
            return

        save_path = DATA_RAW / "political_discourse_2018_2022.parquet"
        
        pl.DataFrame(raw_data)\
            .with_columns([
                pl.col("date").cast(pl.Datetime),
                pl.col("author").cast(pl.Categorical),
                pl.col("text").str.strip_chars()
            ])\
            .sort("date")\
            .write_parquet(save_path)

        logger.info(f"Dataset saved: {save_path}")

if __name__ == "__main__":
    asyncio.run(main())