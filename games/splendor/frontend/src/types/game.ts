export type TokenColor = 'white' | 'blue' | 'green' | 'red' | 'black' | 'gold';

export interface Tokens {
  white: number;
  blue: number;
  green: number;
  red: number;
  black: number;
  gold: number;
}

export interface Card {
  id: number;
  level: number;
  points: number;
  cost: Partial<Tokens>;
  bonus: TokenColor;
}

export interface Noble {
  id: number;
  points: number;
  requirements: Partial<Record<TokenColor, number>>;
}

export interface Player {
  id: number;
  points: number;
  tokens: Partial<Tokens>;
  cards: Card[];
  reservedCards: Card[];
}

export interface GameState {
  players: Player[];
  current_player: number;
  tokens: Tokens;
  available_cards: Record<string, Card[]>;
  available_nobles: Noble[];
}
