import React, { useState } from 'react';
import { GameState, Tokens, TokenColor, Card as CardType, Noble as NobleType, Player } from '../../types/game';
import { Card } from './Card';
import { TokenArea } from './TokenArea';
import { PlayerBoard } from './PlayerBoard';
import { Noble } from './Noble';

interface GameBoardProps {
  gameState: GameState;
  onTakeTokens: (tokens: Partial<Tokens>) => void;
  onBuyCard: (cardId: number) => void;
  onReserveCard: (cardId: number) => void;
  onAcquireNoble: (nobleId: number) => void;
}

export const GameBoard: React.FC<GameBoardProps> = ({
  gameState,
  onTakeTokens,
  onBuyCard,
  onReserveCard,
  onAcquireNoble,
}) => {
  const [selectedTokens, setSelectedTokens] = useState<Partial<Tokens>>({});
  
  const handleTokenSelect = (color: TokenColor) => {
    setSelectedTokens(prev => {
      const current = prev[color] || 0;
      const total = Object.values(prev).reduce((sum, count) => sum + (count || 0), 0);
      const uniqueColors = Object.keys(prev).length;
      
      // If we have 2 of the same color and click it again, deselect both
      if (current === 2) {
        return {};
      }
      
      // If we have 1 of this color and click it again with 4+ available
      if (current === 1 && gameState.tokens[color] >= 4 && uniqueColors === 1) {
        return { [color]: 2 };
      }
      
      // If we have 1 of this color and click it again, deselect it
      if (current === 1) {
        const { [color]: _, ...rest } = prev;
        return rest;
      }
      
      // If we're selecting a new color
      if (current === 0) {
        // If we already have 3 tokens total, can't select more
        if (total >= 3) {
          return prev;
        }
        
        // Allow selecting a new color if we have less than 3 tokens total
        return { ...prev, [color]: 1 };
      }
      
      return prev;
    });
  };

  const handleConfirmTokens = () => {
    onTakeTokens(selectedTokens);
    setSelectedTokens({});
  };

  const getPlayerBonuses = (player: Player): Record<TokenColor, number> => {
    return player.cards.reduce((acc: Record<TokenColor, number>, card: CardType) => {
      const bonus = card.bonus as TokenColor;
      acc[bonus] = (acc[bonus] || 0) + 1;
      return acc;
    }, {
      white: 0,
      blue: 0,
      green: 0,
      red: 0,
      black: 0,
      gold: 0
    });
  };

  const canBuyCard = (card: CardType): boolean => {
    const currentPlayer = gameState.players[gameState.current_player];
    const playerBonuses = getPlayerBonuses(currentPlayer);
    const goldTokensNeeded = Object.entries(card.cost).reduce((total, [color, amount]) => {
      const tokenColor = color as TokenColor;
      const tokenCount = currentPlayer.tokens[tokenColor] || 0;
      const bonusCount = playerBonuses[tokenColor] || 0;
      const needed = Math.max(0, (amount || 0) - tokenCount - bonusCount);
      return total + needed;
    }, 0);
    return goldTokensNeeded <= (currentPlayer.tokens.gold || 0);
  };

  const checkNobleRequirements = (player: Player, noble: NobleType): boolean => {
    const playerBonuses = getPlayerBonuses(player);
    return Object.entries(noble.requirements).every(([color, amount]) => {
      const bonusCount = playerBonuses[color as TokenColor] || 0;
      return bonusCount >= (amount || 0);
    });
  };

  const checkWinCondition = (player: Player): boolean => {
    return player.points >= 15;
  };

  return (
    <div className="container mx-auto p-4 space-y-6">
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold">Splendor</h1>
        <div className="mt-2 text-lg">
          <span className="font-medium">Current Turn: </span>
          <span className="text-blue-600">Player {gameState.current_player + 1}</span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Available Cards</h2>
          {Object.entries(gameState.available_cards || {}).map(([level, cards]) => (
            <div key={level} className="space-y-2">
              <h3 className="text-lg font-medium">Level {level}</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2 gap-2">
                {cards.map((card) => (
                  <Card
                    key={card.id}
                    card={card}
                    onBuy={() => onBuyCard(card.id)}
                    onReserve={() => onReserveCard(card.id)}
                    canBuy={canBuyCard(card)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="space-y-4">
          <div>
            <h2 className="text-xl font-semibold mb-2">Available Nobles</h2>
            <div className="grid grid-cols-2 gap-2">
              {(gameState.available_nobles || []).map((noble) => (
                <Noble key={noble.id} noble={noble} />
              ))}
            </div>
          </div>
          
          <div>
            <h2 className="text-xl font-semibold mb-2">Tokens</h2>
            <TokenArea
              availableTokens={gameState.tokens}
              selectedTokens={selectedTokens}
              onTokenSelect={handleTokenSelect}
              onConfirmSelection={handleConfirmTokens}
            />
          </div>
        </div>

        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Players</h2>
          {gameState.players.map((player, index) => (
            <PlayerBoard
              key={player.id}
              player={player}
              isCurrentPlayer={index === gameState.current_player}
              canBuyCard={canBuyCard}
              onBuyCard={onBuyCard}
              onReserveCard={onReserveCard}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
