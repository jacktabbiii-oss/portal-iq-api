"""
WebSocket Manager for Portal IQ Real-Time Updates

Manages WebSocket connections and broadcasts updates to connected clients
for real-time data synchronization.
"""

import logging
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("portal_iq_api.websocket")


class ConnectionManager:
    """Manages WebSocket connections by channel."""

    def __init__(self):
        # Map of channel -> set of connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)
        logger.info(f"WebSocket connected to channel '{channel}' (total: {len(self.active_connections[channel])})")

    def disconnect(self, websocket: WebSocket, channel: str):
        """Remove a WebSocket connection."""
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
            logger.info(f"WebSocket disconnected from channel '{channel}' (remaining: {len(self.active_connections[channel])})")
            # Clean up empty channels
            if not self.active_connections[channel]:
                del self.active_connections[channel]

    async def broadcast(self, channel: str, message: dict):
        """Broadcast a message to all connections in a channel."""
        if channel not in self.active_connections:
            return

        disconnected = set()
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections[channel].discard(conn)

    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send a message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to WebSocket: {e}")

    def get_channel_count(self, channel: str) -> int:
        """Get the number of connections in a channel."""
        return len(self.active_connections.get(channel, set()))

    def get_total_connections(self) -> int:
        """Get total number of connections across all channels."""
        return sum(len(conns) for conns in self.active_connections.values())


# Singleton instance
manager = ConnectionManager()


# Channel names for different data types
class Channels:
    NIL = "nil"
    PORTAL = "portal"
    DRAFT = "draft"
    ROSTER = "roster"
    ALL = "all"  # Receives all updates


# Message types for different events
class MessageTypes:
    NIL_UPDATE = "nil_update"
    PORTAL_ENTRY = "portal_entry"
    PORTAL_COMMIT = "portal_commit"
    FLIGHT_RISK_CHANGE = "flight_risk_change"
    DRAFT_PROJECTION = "draft_projection"
    ROSTER_CHANGE = "roster_change"
    MARKET_UPDATE = "market_update"


async def notify_nil_update(player_id: str, new_value: float, tier: str):
    """Notify clients of an NIL value update."""
    message = {
        "type": MessageTypes.NIL_UPDATE,
        "player_id": player_id,
        "new_value": new_value,
        "tier": tier,
    }
    await manager.broadcast(Channels.NIL, message)
    await manager.broadcast(Channels.ALL, message)


async def notify_portal_entry(player_name: str, school: str, position: str):
    """Notify clients of a new portal entry."""
    message = {
        "type": MessageTypes.PORTAL_ENTRY,
        "player_name": player_name,
        "school": school,
        "position": position,
    }
    await manager.broadcast(Channels.PORTAL, message)
    await manager.broadcast(Channels.ALL, message)


async def notify_market_update():
    """Notify clients that market data has been refreshed."""
    message = {
        "type": MessageTypes.MARKET_UPDATE,
    }
    await manager.broadcast(Channels.NIL, message)
    await manager.broadcast(Channels.ALL, message)
