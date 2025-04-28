from dotenv import load_dotenv
import os
import re
import logging
from telethon import TelegramClient, events

# Load environment variables
load_dotenv()
api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
session_name = 'igachu'
ALLOWED_CHAT_ID = int(os.getenv("ALLOWED_CHAT_ID"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger("telethon").setLevel(logging.WARNING)

client = TelegramClient(session_name, api_id, api_hash)

def parse_trade_message(message):
    pair_match = re.search(r"([A-Z]{3}/[A-Z]{3})", message)
    otc = "OTC" in message.upper()
    time_match = re.search(r"Entry at (\d{2}:\d{2})", message)
    direction_match = re.search(r"(BUY|SELL)", message.upper())

    currency_pair = None
    if pair_match:
        base_pair = pair_match.group(1).replace("/", "")
        currency_pair = f"{base_pair}_otc" if otc else base_pair

    return {
        "currency_pair": currency_pair,
        "entry_time": time_match.group(1) if time_match else None,
        "trade_direction": "put" if direction_match and direction_match.group(1) == "SELL" else "call" if direction_match else None
    }

@client.on(events.NewMessage)
async def handle_trade_signal(event):
    if event.chat_id != ALLOWED_CHAT_ID:
        return

    message = event.raw_text.strip()
    if "Entry" not in message:
        return

    logging.info("📩 Message received.")
    data = parse_trade_message(message)
    logging.info(f"📊 {data['currency_pair']} | 🕒 {data['entry_time']} | 📈 {data['trade_direction']}")

logging.info("📡 Listening for relevant signals ...")
client.start()
client.run_until_disconnected()
