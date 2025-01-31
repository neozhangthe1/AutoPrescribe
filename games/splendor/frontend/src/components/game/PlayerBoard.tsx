import React from 'react';
import { Player, Card as CardType, TokenColor } from '../../types/game';
import { Card } from './Card';

interface PlayerBoardProps {
  player: Player;
  isCurrentPlayer: boolean;
  canBuyCard: (card: CardType) => boolean;
  onBuyCard: (cardId: number) => void;
  onReserveCard: (cardId: number) => void;
}

export const PlayerBoard: React.FC<PlayerBoardProps> = ({
  player,
  isCurrentPlayer,
  canBuyCard,
  onBuyCard,
  onReserveCard,
}) => {
  const getTokenColorClass = (color: TokenColor): string => {
    const colorMap: Record<TokenColor, string> = {
      white: 'bg-gray-200',
      blue: 'bg-blue-500',
      green: 'bg-green-500',
      red: 'bg-red-500',
      black: 'bg-gray-800',
      gold: 'bg-yellow-500'
    };
    return colorMap[color];
  };

  return (
    <div className={`p-4 rounded-lg ${isCurrentPlayer ? 'bg-blue-50 border-2 border-blue-500' : 'bg-white border'}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Player {player.id + 1}</h3>
        <span className="text-xl font-bold">{player.points} Points</span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="font-medium mb-2">Tokens</h4>
          <div className="space-y-1">
            {Object.entries(player.tokens).map(([color, count]) => (
              count > 0 && (
                <div key={color} className="flex items-center justify-between">
                  <div className={`w-4 h-4 rounded-full ${getTokenColorClass(color as TokenColor)}`} />
                  <span>{count}</span>
                </div>
              )
            ))}
          </div>
        </div>

        <div>
          <h4 className="font-medium mb-2">Bonuses</h4>
          <div className="space-y-1">
            {Object.entries(
              player.cards.reduce((acc: Record<TokenColor, number>, card) => {
                const bonus = card.bonus as TokenColor;
                acc[bonus] = (acc[bonus] || 0) + 1;
                return acc;
              }, {} as Record<TokenColor, number>)
            ).map(([color, count]) => (
              <div key={color} className="flex items-center justify-between">
                <div className={`w-4 h-4 rounded-full ${getTokenColorClass(color as TokenColor)}`} />
                <span>{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {player.reservedCards.length > 0 && (
        <div className="mt-4">
          <h4 className="font-medium mb-2">Reserved Cards</h4>
          <div className="grid grid-cols-2 gap-2">
            {player.reservedCards.map((card) => (
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
      )}
    </div>
  );
};
