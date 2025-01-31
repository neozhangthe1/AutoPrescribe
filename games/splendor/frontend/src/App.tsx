import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { GameBoard } from './components/game/GameBoard';
import { GameState, Tokens } from './types/game';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

const App: React.FC = () => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    startGame();
  }, []);

  const startGame = async () => {
    try {
      const response = await axios.post(`${API_URL}/game/start`, { num_players: 2 });
      setGameState(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to start game. Please try again.');
      console.error('Error starting game:', err);
    }
  };

  const handleTakeTokens = async (tokens: Partial<Tokens>) => {
    if (!gameState) return;
    try {
      const response = await axios.post(
        `${API_URL}/game/take-tokens/${gameState.current_player}`,
        tokens
      );
      setGameState(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to take tokens. Please try again.');
      console.error('Error taking tokens:', err);
    }
  };

  const handleBuyCard = async (cardId: number) => {
    if (!gameState) return;
    try {
      const response = await axios.post(
        `${API_URL}/game/buy-card/${gameState.current_player}/${cardId}`
      );
      setGameState(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to buy card. Please try again.');
      console.error('Error buying card:', err);
    }
  };

  const handleReserveCard = async (cardId: number) => {
    if (!gameState) return;
    try {
      const response = await axios.post(
        `${API_URL}/game/reserve-card/${gameState.current_player}/${cardId}`
      );
      setGameState(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to reserve card. Please try again.');
      console.error('Error reserving card:', err);
    }
  };

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-md">
          <p className="text-red-500 mb-4">{error}</p>
          <button
            onClick={startGame}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!gameState) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <GameBoard
        gameState={gameState}
        onTakeTokens={handleTakeTokens}
        onBuyCard={handleBuyCard}
        onReserveCard={handleReserveCard}
        onAcquireNoble={() => {}}
      />
    </div>
  );
};

export default App;
