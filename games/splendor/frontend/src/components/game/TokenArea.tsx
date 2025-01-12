import React from 'react';
import { Tokens, TokenColor } from '../../types/game';

interface TokenAreaProps {
  availableTokens: Tokens;
  selectedTokens: Partial<Tokens>;
  onTokenSelect: (color: TokenColor) => void;
  onConfirmSelection: () => void;
}

export const TokenArea: React.FC<TokenAreaProps> = ({
  availableTokens,
  selectedTokens,
  onTokenSelect,
  onConfirmSelection
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

  const isValidSelection = () => {
    const selectedCount = Object.values(selectedTokens).reduce((sum: number, count) => sum + (count || 0), 0);
    const uniqueColors = Object.keys(selectedTokens).length;

    // Rule 1: Take exactly 3 different colored tokens
    if (selectedCount === 3 && uniqueColors === 3) return true;

    // Rule 2: Take exactly 2 tokens of the same color (if 4+ available)
    if (selectedCount === 2 && uniqueColors === 1) {
      const color = Object.keys(selectedTokens)[0] as TokenColor;
      return availableTokens[color] >= 4;
    }

    return false;
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-6 gap-4 p-4 bg-gray-100 rounded-lg">
        {(Object.entries(availableTokens) as [TokenColor, number][]).map(([color, count]) => (
          <div key={color} className="flex flex-col items-center gap-2">
            <button
              onClick={() => onTokenSelect(color)}
              disabled={count === 0}
              className={`w-12 h-12 rounded-full transition-all duration-200
                ${count > 0 ? `${getTokenColorClass(color)} hover:opacity-90` : 'bg-gray-300'}
                ${selectedTokens[color] ? 'ring-2 ring-blue-500' : ''}
                flex items-center justify-center text-white font-bold`}
            >
              {count}
            </button>
            <div className="flex flex-col items-center">
              <span className="text-sm capitalize">{color}</span>
              {selectedTokens[color] && (
                <span className="text-xs text-blue-600">Selected: {selectedTokens[color]}</span>
              )}
            </div>
          </div>
        ))}
      </div>
      <button
        onClick={onConfirmSelection}
        disabled={!isValidSelection()}
        className={`w-full py-2 rounded-lg ${
          isValidSelection() ? 'bg-blue-500 hover:bg-blue-600' : 'bg-gray-300'
        } text-white font-semibold transition-colors`}
      >
        Confirm Selection
      </button>
    </div>
  );
};
