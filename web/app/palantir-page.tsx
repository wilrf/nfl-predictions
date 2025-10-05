"use client"

import { useEffect, useState } from "react"
import { PalantirStatCard } from "@/components/ui/palantir-stat-card"
import { PalantirPanel } from "@/components/ui/palantir-panel"
import { PalantirButton } from "@/components/ui/palantir-button"
import { motion, AnimatePresence } from "framer-motion"
import {
  TrendingUp, Target, BarChart3, Zap, Database, Activity,
  Globe, Shield, Terminal, ChevronRight, Search, Command
} from "lucide-react"

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

// Mock data for sparklines
const generateSparkline = (length: number = 10) => {
  return Array.from({ length }, () => Math.random() * 100)
}

export default function PalantirDashboard() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [commandOpen, setCommandOpen] = useState(false)

  // Keyboard shortcut for command palette
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setCommandOpen(true)
      }
      if (e.key === 'Escape') {
        setCommandOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Simulated data fetch (replace with your actual fetch)
  useEffect(() => {
    setTimeout(() => {
      setStats({
        total_games: 256,
        spread_accuracy: 0.573,
        total_accuracy: 0.542,
        spread_correct: 147,
        total_correct: 139,
        high_confidence_count: 48,
        high_confidence_accuracy: 0.812
      })
      setLoading(false)
    }, 1500)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-deep-black">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center space-y-6"
        >
          {/* Futuristic Loading Animation */}
          <div className="relative w-32 h-32 mx-auto">
            <div className="absolute inset-0 border-2 border-electric-blue/20 rounded-full" />
            <div className="absolute inset-0 border-2 border-electric-blue rounded-full animate-ping" />
            <div className="absolute inset-2 border-2 border-cyan/30 rounded-full animate-spin" />
            <div className="absolute inset-4 border-2 border-deep-blue/40 rounded-full animate-pulse" />
            <div className="absolute inset-0 flex items-center justify-center">
              <Database className="w-8 h-8 text-electric-blue animate-pulse" />
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-text-primary text-xl font-display">INITIALIZING SYSTEM</p>
            <p className="text-text-tertiary text-sm font-mono">Establishing secure connection...</p>
          </div>
        </motion.div>
      </div>
    )
  }

  return (
    <main className="min-h-screen bg-deep-black overflow-hidden">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-br from-deep-blue/10 via-transparent to-electric-blue/5" />
        <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-5" />
      </div>

      {/* Navigation Bar */}
      <nav className="relative z-50 border-b border-electric-blue/10 bg-charcoal/80 backdrop-blur-xl">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo Section */}
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-electric-blue to-deep-blue flex items-center justify-center">
                  <Shield className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-text-primary font-display">NFL ANALYTICS</h1>
                  <p className="text-micro text-text-tertiary uppercase tracking-widest">Palantir-Class System</p>
                </div>
              </div>

              {/* Nav Links */}
              <div className="flex items-center gap-1">
                {['Dashboard', 'Predictions', 'Analytics', 'Models'].map((item) => (
                  <button
                    key={item}
                    className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-white/5 rounded-lg transition-all"
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            {/* Right Section */}
            <div className="flex items-center gap-4">
              {/* Command Palette Trigger */}
              <button
                onClick={() => setCommandOpen(true)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-dark-slate/50 border border-electric-blue/20 text-text-secondary hover:text-text-primary hover:border-electric-blue/40 transition-all"
              >
                <Search className="w-4 h-4" />
                <span className="text-xs">Search</span>
                <kbd className="ml-2 px-1.5 py-0.5 text-micro bg-charcoal rounded border border-electric-blue/10">⌘K</kbd>
              </button>

              {/* Status Indicators */}
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
                  <span className="text-micro text-text-tertiary">SYSTEM ONLINE</span>
                </div>
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-electric-blue animate-pulse" />
                  <span className="text-micro text-text-tertiary">LIVE DATA</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="relative z-10 p-6">
        <div className="max-w-[1920px] mx-auto space-y-6">

          {/* Header Section */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-8"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-electric-blue/10 border border-electric-blue/20 mb-4">
              <Terminal className="w-4 h-4 text-electric-blue" />
              <span className="text-sm text-electric-blue font-mono">v4.2.1 QUANTUM</span>
            </div>
            <h1 className="text-6xl font-bold text-text-primary font-display tracking-tight mb-2">
              NFL PREDICTION MATRIX
            </h1>
            <p className="text-lg text-text-secondary">
              Advanced Machine Learning • Real-time Analytics • Quantum Computing
            </p>
          </motion.div>

          {/* Stats Grid */}
          {stats && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <PalantirStatCard
                title="Total Operations"
                value={stats.total_games}
                subtitle="Active predictions analyzed"
                icon={Database}
                trend="up"
                delay={0}
                sparklineData={generateSparkline()}
              />
              <PalantirStatCard
                title="Spread Accuracy"
                value={`${(stats.spread_accuracy * 100).toFixed(1)}%`}
                subtitle={`${stats.spread_correct}/${stats.total_games} correct`}
                icon={Target}
                trend="up"
                status="success"
                delay={0.1}
                sparklineData={generateSparkline()}
              />
              <PalantirStatCard
                title="Total Accuracy"
                value={`${(stats.total_accuracy * 100).toFixed(1)}%`}
                subtitle={`${stats.total_correct}/${stats.total_games} correct`}
                icon={TrendingUp}
                trend="neutral"
                status="warning"
                delay={0.2}
                sparklineData={generateSparkline()}
              />
              <PalantirStatCard
                title="High Confidence"
                value={`${(stats.high_confidence_accuracy * 100).toFixed(1)}%`}
                subtitle={`${stats.high_confidence_count} critical operations`}
                icon={Zap}
                trend="up"
                status="success"
                delay={0.3}
                sparklineData={generateSparkline()}
              />
            </div>
          )}

          {/* Data Panels Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Main Analytics Panel */}
            <div className="lg:col-span-2">
              <PalantirPanel
                title="REAL-TIME PREDICTION MATRIX"
                subtitle="Live game analysis and probability calculations"
                status="live"
                collapsible
              >
                <div className="h-96 flex items-center justify-center text-text-tertiary">
                  {/* Placeholder for main visualization */}
                  <div className="text-center space-y-4">
                    <BarChart3 className="w-16 h-16 mx-auto opacity-20" />
                    <p>Advanced visualization component</p>
                  </div>
                </div>
              </PalantirPanel>
            </div>

            {/* Side Panels */}
            <div className="space-y-4">
              <PalantirPanel
                title="SYSTEM DIAGNOSTICS"
                subtitle="Performance metrics"
                status="idle"
              >
                <div className="space-y-3">
                  {['Model Confidence', 'Data Quality', 'API Latency', 'Cache Hit Rate'].map((metric, i) => (
                    <div key={metric} className="flex items-center justify-between">
                      <span className="text-sm text-text-secondary">{metric}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-1.5 bg-dark-slate rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-electric-blue to-cyan rounded-full"
                            style={{ width: `${60 + i * 10}%` }}
                          />
                        </div>
                        <span className="text-xs text-text-tertiary font-mono">{60 + i * 10}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </PalantirPanel>

              <PalantirPanel
                title="THREAT ASSESSMENT"
                subtitle="Risk analysis"
                status="loading"
              >
                <div className="space-y-2">
                  {['High Risk Games', 'Model Divergence', 'Data Anomalies'].map((threat, i) => (
                    <div key={threat} className="p-3 rounded-lg bg-dark-slate/50 border border-critical/20">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-text-secondary">{threat}</span>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          i === 0 ? 'bg-critical/20 text-critical' : 'bg-warning/20 text-warning'
                        }`}>
                          {i === 0 ? 'CRITICAL' : 'WARNING'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </PalantirPanel>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-center gap-4">
            <PalantirButton variant="primary" icon={Globe} glow>
              Deploy Predictions
            </PalantirButton>
            <PalantirButton variant="secondary" icon={ChevronRight}>
              Run Analysis
            </PalantirButton>
            <PalantirButton variant="ghost" icon={Terminal}>
              Console
            </PalantirButton>
          </div>

        </div>
      </div>

      {/* Command Palette Overlay */}
      <AnimatePresence>
        {commandOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-start justify-center pt-32"
            onClick={() => setCommandOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-2xl bg-dark-slate/95 backdrop-blur-xl rounded-lg border border-electric-blue/20 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 px-4 py-3 border-b border-electric-blue/10">
                <Command className="w-5 h-5 text-electric-blue" />
                <input
                  type="text"
                  placeholder="Search predictions, teams, or run commands..."
                  className="flex-1 bg-transparent text-text-primary placeholder-text-tertiary outline-none"
                  autoFocus
                />
                <kbd className="px-2 py-1 text-micro bg-charcoal rounded border border-electric-blue/10">ESC</kbd>
              </div>
              <div className="p-4">
                <p className="text-sm text-text-tertiary">AI-powered search coming soon...</p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  )
}