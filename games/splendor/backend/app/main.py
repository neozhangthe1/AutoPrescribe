from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from .models import GameState, Tokens, Card
from .game_logic import GameLogic
from typing import Dict
import uvicorn

app = FastAPI(title="Splendor Game API")
game = GameLogic()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080)

@app.post("/game/start", response_model=GameState)
async def start_game(data: dict = Body(..., example={"num_players": 2})):
    """Start a new game with the specified number of players."""
    try:
        num_players = data.get("num_players")
        if num_players is None:
            raise ValueError("Request must include 'num_players' field")
        if not isinstance(num_players, int) or not 2 <= num_players <= 4:
            raise ValueError("Number of players must be between 2 and 4")
        return game.initialize_game(num_players)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/game/take-tokens/{player_id}", response_model=GameState)
async def take_tokens(
    player_id: int,
    tokens: Dict[str, int] = Body(..., examples=[{"white": 1, "blue": 1, "green": 1}])
):
    """Take tokens from the supply."""
    try:
        return game.take_tokens(player_id, tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/game/buy-card/{player_id}/{card_id}", response_model=GameState)
async def buy_card(player_id: int, card_id: int):
    """Buy a development card using tokens and bonuses."""
    try:
        return game.buy_card(player_id, card_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/game/reserve-card/{player_id}/{card_id}", response_model=GameState)
async def reserve_card(player_id: int, card_id: int):
    """Reserve a card and take a gold token if available."""
    try:
        return game.reserve_card(player_id, card_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/game/state", response_model=GameState)
async def get_game_state():
    """Get the current game state."""
    if not game.game_state:
        raise HTTPException(status_code=404, detail="Game not started")
    return game.game_state
