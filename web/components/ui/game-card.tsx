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
  // Defensive checks for missing data
  if (!game.spread_prediction) {
    return (
      <Card className="overflow-hidden">
        <div className="p-4 text-center text-gray-500">
          Loading game data...
        </div>
      </Card>
    )
  }

  const homeWin = game.spread_prediction.predicted_winner === game.home_team
  const probability = homeWin ? game.spread_prediction.home_win_prob : game.spread_prediction.away_win_prob

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      whileHover={{ scale: 1.01, y: -4 }}
      className="group"
    >
      <Card className="overflow-hidden border border-white/5 shadow-xl hover:shadow-2xl hover:shadow-white/10 transition-all duration-500">
        {/* Subtle gradient overlay on hover */}
        <div className="absolute inset-0 bg-gradient-to-r from-white/[0.03] via-transparent to-white/[0.03] opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

        <div className="relative grid grid-cols-3 items-center gap-6">
          {/* Away Team */}
          <div className="text-center space-y-2 transition-transform duration-300 group-hover:scale-105">
            <div className="text-lg font-bold text-white uppercase tracking-wide group-hover:text-white/90 transition-colors">
              {game.away_team}
            </div>
            {game.away_score !== undefined && (
              <div className="text-4xl font-bold text-white tabular-nums">
                {game.away_score}
              </div>
            )}
          </div>

          {/* Prediction */}
          <div className="text-center space-y-3 py-4 px-6 glass rounded-xl relative group-hover:bg-white/[0.08] transition-all duration-300">
            {/* Top accent line */}
            <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

            <div className="text-sm text-medium-gray uppercase tracking-wider group-hover:text-white/70 transition-colors">
              Predicted Winner
            </div>
            <div className="text-lg font-bold text-white group-hover:scale-105 transition-transform duration-300">
              {game.spread_prediction.predicted_winner}
            </div>

            {/* Probability Bar */}
            <div className="relative h-8 bg-charcoal rounded-full overflow-hidden shadow-inner">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${probability * 100}%` }}
                transition={{ duration: 1, delay: index * 0.05 + 0.2, ease: "easeOut" }}
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-white via-white to-light-gray flex items-center justify-center shadow-lg group-hover:shadow-white/30 transition-shadow duration-300"
              >
                <span className="text-xs font-bold text-black px-2 tabular-nums">
                  {(probability * 100).toFixed(0)}%
                </span>
              </motion.div>
            </div>

            {/* Result */}
            <div className="flex items-center justify-center gap-2 mt-3">
              {game.spread_prediction.correct ? (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 group-hover:bg-green-500/20 transition-colors duration-300">
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                  <span className="text-sm font-semibold text-green-400">Correct</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-500/10 group-hover:bg-red-500/20 transition-colors duration-300">
                  <XCircle className="w-4 h-4 text-red-400" />
                  <span className="text-sm font-semibold text-red-400">Incorrect</span>
                </div>
              )}
            </div>
          </div>

          {/* Home Team */}
          <div className="text-center space-y-2 transition-transform duration-300 group-hover:scale-105">
            <div className="text-lg font-bold text-white uppercase tracking-wide group-hover:text-white/90 transition-colors">
              {game.home_team}
            </div>
            {game.home_score !== undefined && (
              <div className="text-4xl font-bold text-white tabular-nums">
                {game.home_score}
              </div>
            )}
          </div>
        </div>
      </Card>
    </motion.div>
  )
}
