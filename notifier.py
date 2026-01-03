
import os
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime

logger = logging.getLogger("Notifier")

class Notifier:
    """
    Handles external notifications via Discord Webhooks.
    Designed to be failsafe - if notification fails, it logs locally and continues.
    """
    def __init__(self, prefix="[CHAMPION]"):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        self.prefix = prefix
        if self.webhook_url:
            logger.info("Discord Webhook detected. Notifications ENABLED.")
        else:
            logger.info("No Discord Webhook found. Notifications DISABLED (Local Log Only).")

    def send(self, message: str, level: str = "INFO"):
        """
        Send a message to Discord.
        Levels: INFO, WARNING, CRITICAL, SUCCESS
        """
        # Always log locally first
        log_msg = f"{level}: {message}"
        if level == "CRITICAL": logger.critical(message)
        elif level == "WARNING": logger.warning(message)
        else: logger.info(message)

        if not self.webhook_url:
            return

        # Map levels to colors (approximate)
        colors = {
            "INFO": 3447003,      # Blue
            "SUCCESS": 5763719,   # Green
            "WARNING": 16776960,  # Yellow
            "CRITICAL": 15548997  # Red
        }
        
        timestamp = datetime.utcnow().isoformat()
        
        payload = {
            "username": "Sofia Champion",
            "embeds": [{
                "title": f"{self.prefix} {level}",
                "description": message,
                "color": colors.get(level, 3447003),
                "timestamp": timestamp,
                "footer": {"text": "Live Trading System"}
            }]
        }

        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'User-Agent': 'SofiaBot/1.0'}
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                pass # Success
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

# Global instance for easy import if needed, 
# but usually instantiated by the bot class.
