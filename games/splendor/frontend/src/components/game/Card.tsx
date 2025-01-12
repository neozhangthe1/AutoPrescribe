import React from 'react';
import { Card as CardType, TokenColor } from '../../types/game';

interface CardProps {
  card: CardType;
  onBuy: () => void;
  onReserve: () => void;
  canBuy: boolean;
}

export const Card: React.FC<CardProps> = ({ card, onBuy, onReserve, canBuy }) => {
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
    <div className="p-4 border rounded-lg shadow-sm bg-white">
      <div className="flex justify-between items-center mb-2">
        <span className={`w-6 h-6 rounded-full ${getTokenColorClass(card.bonus)}`} />
        <span className="font-bold">{card.points}</span>
      </div>
      <div className="space-y-1">
        {Object.entries(card.cost).map(([color, count]) => (
          <div key={color} className="flex items-center justify-between">
            <span className={`w-4 h-4 rounded-full ${getTokenColorClass(color as TokenColor)}`} />
            <span>{count}</span>
          </div>
        ))}
      </div>
      <div className="mt-4 space-y-2">
        <button
          onClick={onBuy}
          disabled={!canBuy}
          className={`w-full py-1 px-2 rounded ${
            canBuy ? 'bg-blue-500 hover:bg-blue-600' : 'bg-gray-300'
          } text-white text-sm`}
        >
          Buy
        </button>
        <button
          onClick={onReserve}
          className="w-full py-1 px-2 rounded bg-gray-200 hover:bg-gray-300 text-sm"
        >
          Reserve
        </button>
      </div>
    </div>
  );
};
