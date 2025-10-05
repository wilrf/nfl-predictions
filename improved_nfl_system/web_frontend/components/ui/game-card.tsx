"use client"

import { Card } from "./card"
import { motion } from "framer-motion"
import { CheckCircle2, XCircle } from "lucide-react"

interface GameCardProps {
  game: {
    game_id: string
    week: number
    away_team: string
    home_team: string
    away_score?: number
    home_score?: number
    spread_prediction: {
      predicted_winner: string
      home_win_prob: number
      away_win_prob: number
      confidence: number
      correct: boolean
    }
    total_prediction: {
      predicted: string
      over_prob: number
      under_prob: number
      confidence: number
      correct: boolean
    }
    actual_winner: string
    total_points?: number
  }
  index: number
}

export function GameCard({ game, index }: GameCardProps) {
  const homeWin = game.spread_prediction.predicted_winner === game.home_team
  const probability = homeWin ? game.spread_prediction.home_win_prob : game.spread_prediction.away_win_prob

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      whileHover={{ scale: 1.02 }}
    >
      <Card className="overflow-hidden">
        <div className="grid grid-cols-3 items-center gap-6">
          {/* Away Team */}
          <div className="text-center space-y-2">
            <div className="text-lg font-bold text-white uppercase tracking-wide">
              {game.away_team}
            </div>
            {game.away_score !== undefined && (
              <div className="text-4xl font-bold text-white">
                {game.away_score}
              </div>
            )}
          </div>

          {/* Prediction */}
          <div className="text-center space-y-3 py-4 px-6 glass rounded-xl">
            <div className="text-sm text-medium-gray uppercase tracking-wider">
              Predicted Winner
            </div>
            <div className="text-lg font-bold text-white">
              {game.spread_prediction.predicted_winner}
            </div>

            {/* Probability Bar */}
            <div className="relative h-8 bg-charcoal rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${probability * 100}%` }}
                transition={{ duration: 1, delay: index * 0.05 + 0.2 }}
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-white to-light-gray flex items-center justify-center"
              >
                <span className="text-xs font-bold text-black px-2">
                  {(probability * 100).toFixed(0)}%
                </span>
              </motion.div>
            </div>

            {/* Result */}
            <div className="flex items-center justify-center gap-2 mt-3">
              {game.spread_prediction.correct ? (
                <>
                  <CheckCircle2 className="w-5 h-5 text-white" />
                  <span className="text-sm font-medium text-white">Correct</span>
                </>
              ) : (
                <>
                  <XCircle className="w-5 h-5 text-medium-gray" />
                  <span className="text-sm font-medium text-medium-gray">Incorrect</span>
                </>
              )}
            </div>
          </div>

          {/* Home Team */}
          <div className="text-center space-y-2">
            <div className="text-lg font-bold text-white uppercase tracking-wide">
              {game.home_team}
            </div>
            {game.home_score !== undefined && (
              <div className="text-4xl font-bold text-white">
                {game.home_score}
              </div>
            )}
          </div>
        </div>
      </Card>
    </motion.div>
  )
}
