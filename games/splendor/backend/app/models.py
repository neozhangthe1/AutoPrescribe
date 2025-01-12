from typing import Dict, List, Optional
from pydantic import BaseModel

class Tokens(BaseModel):
    white: int = 0
    blue: int = 0
    green: int = 0
    red: int = 0
    black: int = 0
    gold: int = 0

class Card(BaseModel):
    id: int
    level: int
    points: int
    cost: Dict[str, int]
    bonus: str

class Noble(BaseModel):
    id: int
    points: int
    requirements: Dict[str, int]

class Player(BaseModel):
    id: int
    points: int = 0
    tokens: Tokens = Tokens()
    cards: List[Card] = []
    reserved_cards: List[Card] = []

class GameState(BaseModel):
    players: List[Player]
    current_player: int
    tokens: Tokens
    available_cards: Dict[str, List[Card]]
    available_nobles: List[Noble]
