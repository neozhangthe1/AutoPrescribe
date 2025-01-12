from typing import Dict, List, Optional
from .models import GameState, Player, Card, Noble, Tokens
import random

class GameLogic:
    def __init__(self):
        self._state: Optional[GameState] = None
        self._cards: Dict[int, List[Card]] = self._initialize_cards()
        self._nobles: List[Noble] = self._initialize_nobles()

    @property
    def game_state(self) -> Optional[GameState]:
        return self._state

    def initialize_game(self, num_players: int) -> GameState:
        if not 2 <= num_players <= 4:
            raise ValueError("Number of players must be between 2 and 4")

        # Initialize tokens based on player count
        token_count = 4 if num_players == 2 else (5 if num_players == 3 else 7)
        tokens = Tokens(
            white=token_count,
            blue=token_count,
            green=token_count,
            red=token_count,
            black=token_count,
            gold=5
        )

        # Initialize players
        players = [Player(id=i) for i in range(num_players)]

        # Initialize available cards
        available_cards = {
            f"level_{level}": self._draw_cards(level, 4)
            for level in range(1, 4)
        }

        # Select nobles based on player count
        num_nobles = num_players + 1
        available_nobles = random.sample(self._nobles, num_nobles)

        self._state = GameState(
            players=players,
            current_player=0,
            tokens=tokens,
            available_cards=available_cards,
            available_nobles=available_nobles
        )

        return self._state

    def take_tokens(self, player_id: int, tokens: Dict[str, int]) -> GameState:
        if not self._state:
            raise ValueError("Game not started")

        if player_id != self._state.current_player:
            raise ValueError("Not your turn")

        # Validate token selection
        selected_colors = len(tokens)
        total_tokens = sum(tokens.values())

        if selected_colors == 1:
            # Taking 2 of the same color
            color = list(tokens.keys())[0]
            if tokens[color] != 2:
                raise ValueError("When taking same color, must take exactly 2")
            if getattr(self._state.tokens, color) < 4:
                raise ValueError("Must have 4+ tokens to take 2 of same color")
        elif selected_colors == 3:
            # Taking 3 different colors
            if any(count != 1 for count in tokens.values()):
                raise ValueError("When taking different colors, must take exactly 1 of each")
        else:
            raise ValueError("Must take either 2 same tokens or 3 different tokens")

        # Update tokens
        player = self._state.players[player_id]
        for color, count in tokens.items():
            current = getattr(self._state.tokens, color)
            if current < count:
                raise ValueError(f"Not enough {color} tokens available")
            setattr(self._state.tokens, color, current - count)
            current_player = getattr(player.tokens, color)
            setattr(player.tokens, color, current_player + count)

        # Move to next player
        self._state.current_player = (self._state.current_player + 1) % len(self._state.players)
        return self._state

    def buy_card(self, player_id: int, card_id: int) -> GameState:
        if not self._state:
            raise ValueError("Game not started")

        if player_id != self._state.current_player:
            raise ValueError("Not your turn")

        # Find the card
        card = None
        for cards in self._state.available_cards.values():
            for c in cards:
                if c.id == card_id:
                    card = c
                    break
            if card:
                break

        if not card:
            # Check reserved cards
            player = self._state.players[player_id]
            for c in player.reserved_cards:
                if c.id == card_id:
                    card = c
                    break

        if not card:
            raise ValueError("Card not found")

        # Check if player can afford the card
        player = self._state.players[player_id]
        required_tokens = self._calculate_required_tokens(player, card)

        # Pay tokens
        for color, count in required_tokens.items():
            if count > 0:
                current = getattr(player.tokens, color)
                if current < count:
                    raise ValueError(f"Not enough {color} tokens")
                setattr(player.tokens, color, current - count)
                current_supply = getattr(self._state.tokens, color)
                setattr(self._state.tokens, color, current_supply + count)

        # Add card to player's collection
        if card in player.reserved_cards:
            player.reserved_cards.remove(card)
        else:
            # Remove from available cards and draw new one
            for level, cards in self._state.available_cards.items():
                if card in cards:
                    cards.remove(card)
                    new_card = self._draw_cards(int(level[-1]), 1)[0]
                    cards.append(new_card)
                    break

        player.cards.append(card)
        player.points += card.points

        # Check for noble acquisition
        self._check_nobles(player)

        # Move to next player
        self._state.current_player = (self._state.current_player + 1) % len(self._state.players)
        return self._state

    def reserve_card(self, player_id: int, card_id: int) -> GameState:
        if not self._state:
            raise ValueError("Game not started")

        if player_id != self._state.current_player:
            raise ValueError("Not your turn")

        player = self._state.players[player_id]
        if len(player.reserved_cards) >= 3:
            raise ValueError("Cannot reserve more than 3 cards")

        # Find and reserve the card
        card = None
        for cards in self._state.available_cards.values():
            for c in cards:
                if c.id == card_id:
                    card = c
                    cards.remove(c)
                    # Draw new card
                    level = card.level
                    new_card = self._draw_cards(level, 1)[0]
                    cards.append(new_card)
                    break
            if card:
                break

        if not card:
            raise ValueError("Card not found")

        player.reserved_cards.append(card)

        # Give gold token if available
        if self._state.tokens.gold > 0:
            self._state.tokens.gold -= 1
            player.tokens.gold += 1

        # Move to next player
        self._state.current_player = (self._state.current_player + 1) % len(self._state.players)
        return self._state

    def _initialize_cards(self) -> Dict[int, List[Card]]:
        # Simplified card initialization for demonstration
        cards: Dict[int, List[Card]] = {1: [], 2: [], 3: []}
        card_id = 1

        # Level 1 cards
        for _ in range(40):
            cost = {random.choice(['white', 'blue', 'green', 'red', 'black']): random.randint(2, 4)}
            cards[1].append(Card(
                id=card_id,
                level=1,
                points=0,
                cost=cost,
                bonus=random.choice(['white', 'blue', 'green', 'red', 'black'])
            ))
            card_id += 1

        # Level 2 cards
        for _ in range(30):
            cost = {
                random.choice(['white', 'blue', 'green', 'red', 'black']): random.randint(3, 5),
                random.choice(['white', 'blue', 'green', 'red', 'black']): random.randint(2, 3)
            }
            cards[2].append(Card(
                id=card_id,
                level=2,
                points=random.randint(1, 2),
                cost=cost,
                bonus=random.choice(['white', 'blue', 'green', 'red', 'black'])
            ))
            card_id += 1

        # Level 3 cards
        for _ in range(20):
            cost = {
                random.choice(['white', 'blue', 'green', 'red', 'black']): random.randint(4, 6),
                random.choice(['white', 'blue', 'green', 'red', 'black']): random.randint(3, 4),
                random.choice(['white', 'blue', 'green', 'red', 'black']): random.randint(2, 3)
            }
            cards[3].append(Card(
                id=card_id,
                level=3,
                points=random.randint(3, 5),
                cost=cost,
                bonus=random.choice(['white', 'blue', 'green', 'red', 'black'])
            ))
            card_id += 1

        return cards

    def _initialize_nobles(self) -> List[Noble]:
        # Simplified noble initialization
        nobles = []
        for i in range(10):
            requirements = {
                random.choice(['white', 'blue', 'green', 'red', 'black']): 4,
                random.choice(['white', 'blue', 'green', 'red', 'black']): 4
            }
            nobles.append(Noble(
                id=i + 1,
                points=3,
                requirements=requirements
            ))
        return nobles

    def _draw_cards(self, level: int, count: int) -> List[Card]:
        if level not in self._cards or not self._cards[level]:
            return []
        cards = random.sample(self._cards[level], min(count, len(self._cards[level])))
        for card in cards:
            self._cards[level].remove(card)
        return cards

    def _calculate_required_tokens(self, player: Player, card: Card) -> Dict[str, int]:
        required = {}
        for color, count in card.cost.items():
            # Calculate bonuses from owned cards
            bonus = sum(1 for c in player.cards if c.bonus == color)
            # Calculate required tokens after applying bonuses
            required[color] = max(0, count - bonus)
        return required

    def _check_nobles(self, player: Player) -> None:
        if not self._state:
            return

        # Get card bonuses
        bonuses = {}
        for card in player.cards:
            bonuses[card.bonus] = bonuses.get(card.bonus, 0) + 1

        # Check each noble
        acquired_nobles = []
        for noble in self._state.available_nobles:
            can_acquire = True
            for color, count in noble.requirements.items():
                if bonuses.get(color, 0) < count:
                    can_acquire = False
                    break
            if can_acquire:
                acquired_nobles.append(noble)
                player.points += noble.points

        # Remove acquired nobles
        for noble in acquired_nobles:
            self._state.available_nobles.remove(noble)
