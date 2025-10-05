"use client"

import { useEffect, useState } from "react"
import { StatCard } from "@/components/ui/stat-card"
import { GameCard } from "@/components/ui/game-card"
import { WeeklyPerformanceChart } from "@/components/charts/weekly-performance-chart"
import { ConfidenceChart } from "@/components/charts/confidence-chart"
import { motion } from "framer-motion"
import { TrendingUp, Target, BarChart3, Zap } from "lucide-react"

interface Stats {
  total_games: number
  spread_accuracy: number
  total_accuracy: number
  spread_correct: number
  total_correct: number
  high_confidence_count: number
  high_confidence_accuracy: number
}

interface Game {
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

interface WeeklyData {
  week: number
  games: number
  spread_accuracy: number
  total_accuracy: number
  spread_correct: number
  total_correct: number
}

interface ConfidenceData {
  bucket: string
  min_confidence: number
  count: number
  accuracy: number
}

// API is on same domain when deployed to Vercel
// For local dev, use localhost:8000
const API_URL = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && window.location.origin.includes('vercel.app') ? '' : 'http://localhost:8000')

export default function Home() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [games, setGames] = useState<Game[]>([])
  const [weeklyData, setWeeklyData] = useState<WeeklyData[]>([])
  const [confidenceData, setConfidenceData] = useState<ConfidenceData[]>([])
  const [selectedWeek, setSelectedWeek] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const [statsRes, gamesRes, weeklyRes, confidenceRes] = await Promise.all([
          fetch(`${API_URL}/api/stats`),
          fetch(`${API_URL}/api/games`),
          fetch(`${API_URL}/api/weekly_performance`),
          fetch(`${API_URL}/api/confidence_analysis`)
        ])

        const statsData = await statsRes.json()
        const gamesData = await gamesRes.json()
        const weeklyDataRes = await weeklyRes.json()
        const confidenceDataRes = await confidenceRes.json()

        setStats(statsData)
        setGames(gamesData)
        setWeeklyData(weeklyDataRes)
        setConfidenceData(confidenceDataRes)

        if (weeklyDataRes.length > 0) {
          setSelectedWeek(weeklyDataRes[0].week)
        }
      } catch (error) {
        console.error("Error fetching data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const weeks = Array.from(new Set(games.map(g => g.week))).sort((a, b) => a - b)
  const filteredGames = selectedWeek
    ? games.filter(g => g.week === selectedWeek)
    : games

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center space-y-4"
        >
          <div className="w-16 h-16 border-4 border-white/20 border-t-white rounded-full animate-spin mx-auto" />
          <p className="text-white text-xl">Loading NFL Predictions...</p>
        </motion.div>
      </div>
    )
  }

  return (
    <main className="min-h-screen p-6 md:p-12">
      <div className="max-w-7xl mx-auto space-y-12">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center space-y-4 py-12"
        >
          <h1 className="text-6xl md:text-7xl font-bold text-white tracking-tight">
            NFL ML Predictions
          </h1>
          <p className="text-xl text-medium-gray">
            XGBoost Machine Learning • Real-time Analytics • Premium Insights
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-medium-gray">
            <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
            <span>Live Dashboard</span>
          </div>
        </motion.header>

        {/* Stats Grid */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatCard
              title="Total Games"
              value={stats.total_games}
              icon={BarChart3}
              delay={0}
            />
            <StatCard
              title="Spread Accuracy"
              value={`${(stats.spread_accuracy * 100).toFixed(1)}%`}
              subtitle={`${stats.spread_correct}/${stats.total_games} correct`}
              icon={Target}
              trend="up"
              delay={0.1}
            />
            <StatCard
              title="Total Accuracy"
              value={`${(stats.total_accuracy * 100).toFixed(1)}%`}
              subtitle={`${stats.total_correct}/${stats.total_games} correct`}
              icon={TrendingUp}
              trend="up"
              delay={0.2}
            />
            <StatCard
              title="High Confidence"
              value={`${(stats.high_confidence_accuracy * 100).toFixed(1)}%`}
              subtitle={`${stats.high_confidence_count} games`}
              icon={Zap}
              trend="up"
              delay={0.3}
            />
          </div>
        )}

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {weeklyData.length > 0 && <WeeklyPerformanceChart data={weeklyData} />}
          {confidenceData.length > 0 && <ConfidenceChart data={confidenceData} />}
        </div>

        {/* Games Section */}
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="space-y-6"
        >
          <h2 className="text-4xl font-bold text-white text-center">
            Game Predictions & Results
          </h2>

          {/* Week Tabs */}
          <div className="flex flex-wrap justify-center gap-3">
            {weeks.map((week) => (
              <motion.button
                key={week}
                onClick={() => setSelectedWeek(week)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                  selectedWeek === week
                    ? 'bg-white text-black'
                    : 'glass text-white hover:bg-white/10'
                }`}
              >
                Week {week}
              </motion.button>
            ))}
          </div>

          {/* Games Grid */}
          <div className="space-y-4">
            {filteredGames.map((game, index) => (
              <GameCard key={game.game_id} game={game} index={index} />
            ))}
          </div>
        </motion.section>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="text-center space-y-2 py-12 text-medium-gray"
        >
          <p className="text-lg">Powered by XGBoost ML • Trained on 2,351+ games (2015-2023)</p>
          <p className="text-sm">Data: nfl_data_py • Models: Isotonic Calibrated Probabilities</p>
        </motion.footer>
      </div>
    </main>
  )
}
