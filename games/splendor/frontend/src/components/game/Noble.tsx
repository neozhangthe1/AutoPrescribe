import React from 'react';
import { Noble as NobleType, TokenColor } from '../../types/game';

interface NobleProps {
  noble: NobleType;
}

export const Noble: React.FC<NobleProps> = ({ noble }) => {
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
    <div className="p-3 border rounded-lg shadow-sm bg-white">
      <div className="text-center font-bold mb-2">{noble.points}</div>
      <div className="space-y-1">
        {Object.entries(noble.requirements).map(([color, count]) => (
          <div key={color} className="flex items-center justify-between">
            <span className={`w-4 h-4 rounded-full ${getTokenColorClass(color as TokenColor)}`} />
            <span>{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
