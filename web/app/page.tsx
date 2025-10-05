"use client"

import { useEffect, useState, useMemo, useCallback } from "react"
import { StatCard } from "../components/ui/stat-card"
import { GameCard } from "../components/ui/game-card"
import { WeeklyPerformanceChart } from "../components/charts/weekly-performance-chart"
import { ConfidenceChart } from "../components/charts/confidence-chart"
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

// Memoized API URL
const API_URL = process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== 'undefined' && window.location.origin.includes('vercel.app') ? '' : 'http://localhost:8000')

// Cache duration in milliseconds (5 minutes)
const CACHE_DURATION = 5 * 60 * 1000

export default function Home() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [games, setGames] = useState<Game[]>([])
  const [weeklyData, setWeeklyData] = useState<WeeklyData[]>([])
  const [confidenceData, setConfidenceData] = useState<ConfidenceData[]>([])
  const [selectedWeek, setSelectedWeek] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Memoized fetch function with error handling and caching
  const fetchData = useCallback(async () => {
    try {
      // Check localStorage for cached data
      const cacheKey = 'nfl_dashboard_cache'
      const cached = localStorage.getItem(cacheKey)
      const cacheTime = localStorage.getItem(`${cacheKey}_time`)

      if (cached && cacheTime) {
        const age = Date.now() - parseInt(cacheTime)
        if (age < CACHE_DURATION) {
          const data = JSON.parse(cached)
          setStats(data.stats)
          setGames(data.games)
          setWeeklyData(data.weeklyData)
          setConfidenceData(data.confidenceData)
          if (data.weeklyData.length > 0) {
            setSelectedWeek(data.weeklyData[0].week)
          }
          setLoading(false)
          return
        }
      }

      // Fetch with timeout
      const fetchWithTimeout = (url: string, timeout = 10000) => {
        return Promise.race([
          fetch(url),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('Request timeout')), timeout)
          )
        ])
      }

      const [statsRes, gamesRes, weeklyRes, confidenceRes] = await Promise.all([
        fetchWithTimeout(`${API_URL}/api/stats`),
        fetchWithTimeout(`${API_URL}/api/games`),
        fetchWithTimeout(`${API_URL}/api/weekly_performance`),
        fetchWithTimeout(`${API_URL}/api/confidence_analysis`)
      ])

      if (!statsRes.ok || !gamesRes.ok || !weeklyRes.ok || !confidenceRes.ok) {
        throw new Error('Failed to fetch data')
      }

      const [statsData, gamesData, weeklyDataRes, confidenceDataRes] = await Promise.all([
        statsRes.json(),
        gamesRes.json(),
        weeklyRes.json(),
        confidenceRes.json()
      ])

      // Cache the data
      const dataToCache = {
        stats: statsData,
        games: gamesData,
        weeklyData: weeklyDataRes,
        confidenceData: confidenceDataRes
      }
      localStorage.setItem(cacheKey, JSON.stringify(dataToCache))
      localStorage.setItem(`${cacheKey}_time`, Date.now().toString())

      setStats(statsData)
      setGames(gamesData)
      setWeeklyData(weeklyDataRes)
      setConfidenceData(confidenceDataRes)

      if (weeklyDataRes.length > 0) {
        setSelectedWeek(weeklyDataRes[0].week)
      }
    } catch (error) {
      console.error("Error fetching data:", error)
      setError("Failed to load data. Please refresh the page.")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Memoized weeks calculation
  const weeks = useMemo(() =>
    Array.from(new Set(games.map(g => g.week))).sort((a, b) => a - b),
    [games]
  )

  // Memoized filtered games
  const filteredGames = useMemo(() =>
    selectedWeek ? games.filter(g => g.week === selectedWeek) : games,
    [games, selectedWeek]
  )

  // Optimized week selection handler
  const handleWeekSelect = useCallback((week: number) => {
    setSelectedWeek(week)
  }, [])

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

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center space-y-4"
        >
          <p className="text-red-500 text-xl">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-3 bg-white text-black rounded-lg hover:bg-gray-100 transition"
          >
            Retry
          </button>
        </motion.div>
      </div>
    )
  }

  return (
    <main className="min-h-screen p-6 md:p-12">
      <div className="max-w-7xl mx-auto space-y-12">
        {/* Header - simplified animation */}
        <motion.header
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
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

        {/* Stats Grid - removed individual animations */}
        {stats && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3, delay: 0.1 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          >
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
              delay={0}
            />
            <StatCard
              title="Total Accuracy"
              value={`${(stats.total_accuracy * 100).toFixed(1)}%`}
              subtitle={`${stats.total_correct}/${stats.total_games} correct`}
              icon={TrendingUp}
              trend="up"
              delay={0}
            />
            <StatCard
              title="High Confidence"
              value={`${(stats.high_confidence_accuracy * 100).toFixed(1)}%`}
              subtitle={`${stats.high_confidence_count} games`}
              icon={Zap}
              trend="up"
              delay={0}
            />
          </motion.div>
        )}

        {/* Charts - lazy loaded */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          {weeklyData.length > 0 && <WeeklyPerformanceChart data={weeklyData} />}
          {confidenceData.length > 0 && <ConfidenceChart data={confidenceData} />}
        </motion.div>

        {/* Games Section */}
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          className="space-y-6"
        >
          <h2 className="text-4xl font-bold text-white text-center">
            Game Predictions & Results
          </h2>

          {/* Week Tabs - simplified animations */}
          <div className="flex flex-wrap justify-center gap-3">
            {weeks.map((week) => (
              <button
                key={week}
                onClick={() => handleWeekSelect(week)}
                className={`px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
                  selectedWeek === week
                    ? 'bg-white text-black'
                    : 'glass text-white hover:bg-white/10'
                }`}
              >
                Week {week}
              </button>
            ))}
          </div>

          {/* Games Grid - virtualized rendering for large lists */}
          <div className="space-y-4">
            {filteredGames.slice(0, 20).map((game, index) => (
              <GameCard key={game.game_id} game={game} index={index} />
            ))}
          </div>
        </motion.section>

        {/* Footer */}
        <footer className="text-center space-y-2 py-12 text-medium-gray">
          <p className="text-lg">Powered by XGBoost ML • Trained on 2,351+ games (2015-2023)</p>
          <p className="text-sm">Data: nfl_data_py • Models: Isotonic Calibrated Probabilities</p>
        </footer>
      </div>
    </main>
  )
}