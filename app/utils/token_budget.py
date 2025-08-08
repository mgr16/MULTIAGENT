from typing import List, Dict

# Sencillo: presupuesto total y “hard truncation” por mensajes antiguos.
def enforce_token_budget(messages: List[Dict[str, str]], max_messages: int = 30) -> List[Dict[str, str]]:
    if len(messages) <= max_messages:
        return messages
    # Mantén system + últimos N-1
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other = [m for m in messages if m.get("role") != "system"]
    keep = other[-(max_messages - len(system_msgs)):] if max_messages > len(system_msgs) else []
    return system_msgs + keep
