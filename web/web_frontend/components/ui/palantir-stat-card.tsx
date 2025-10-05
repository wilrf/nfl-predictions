"use client"

import { motion } from "framer-motion"
import { LucideIcon } from "lucide-react"
import { useMemo } from "react"

interface PalantirStatCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon?: LucideIcon
  trend?: "up" | "down" | "neutral"
  status?: "default" | "success" | "warning" | "critical"
  delay?: number
  sparklineData?: number[]
}

export function PalantirStatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  status = "default",
  delay = 0,
  sparklineData
}: PalantirStatCardProps) {

  const statusColors = {
    default: "border-electric-blue/20 shadow-glow",
    success: "border-success/30 shadow-[0_0_20px_rgba(0,255,136,0.3)]",
    warning: "border-warning/30 shadow-[0_0_20px_rgba(255,184,0,0.3)]",
    critical: "border-critical/30 shadow-[0_0_20px_rgba(255,51,102,0.3)]"
  }

  const trendIcons = {
    up: "↗",
    down: "↘",
    neutral: "→"
  }

  const trendColors = {
    up: "text-success",
    down: "text-critical",
    neutral: "text-text-secondary"
  }

  // Generate sparkline path if data is provided
  const sparklinePath = useMemo(() => {
    if (!sparklineData || sparklineData.length < 2) return null

    const width = 100
    const height = 30
    const max = Math.max(...sparklineData)
    const min = Math.min(...sparklineData)
    const range = max - min || 1

    const points = sparklineData.map((value, index) => {
      const x = (index / (sparklineData.length - 1)) * width
      const y = height - ((value - min) / range) * height
      return `${x},${y}`
    }).join(' ')

    return `M ${points}`
  }, [sparklineData])

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        duration: 0.5,
        delay,
        ease: [0.16, 1, 0.3, 1]
      }}
      whileHover={{
        y: -2,
        transition: { duration: 0.2 }
      }}
      className="relative group"
    >
      {/* Main Card Container */}
      <div className={`
        relative overflow-hidden rounded-lg
        bg-gradient-to-br from-dark-slate/90 to-charcoal/90
        backdrop-blur-xl border ${statusColors[status]}
        transition-all duration-300
        group-hover:shadow-glow-intense
      `}>

        {/* Animated Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0 bg-grid-pattern bg-grid animate-pulse" />
        </div>

        {/* Scan Line Effect */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-electric-blue to-transparent animate-scan-line" />
        </div>

        {/* Content Container */}
        <div className="relative z-10 p-6">

          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <div className="text-micro uppercase tracking-[0.08em] text-text-tertiary font-medium mb-1">
                {title}
              </div>
              {/* Live Indicator */}
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-electric-blue rounded-full animate-pulse" />
                <span className="text-micro text-text-tertiary">LIVE</span>
              </div>
            </div>

            {Icon && (
              <div className={`
                p-2.5 rounded-lg
                bg-gradient-to-br from-electric-blue/10 to-deep-blue/10
                border border-electric-blue/20
                group-hover:border-electric-blue/40
                transition-all duration-300
              `}>
                <Icon className="w-5 h-5 text-electric-blue" />
              </div>
            )}
          </div>

          {/* Main Value */}
          <div className="mb-4">
            <div className="text-4xl font-bold text-text-primary tracking-tight font-display">
              {value}
            </div>

            {/* Trend Indicator */}
            {trend && (
              <div className={`inline-flex items-center gap-1 mt-2 ${trendColors[trend]}`}>
                <span className="text-lg">{trendIcons[trend]}</span>
                <span className="text-xs font-medium">
                  {trend === "up" ? "Increasing" : trend === "down" ? "Decreasing" : "Stable"}
                </span>
              </div>
            )}
          </div>

          {/* Sparkline Visualization */}
          {sparklinePath && (
            <div className="mb-3 h-8 relative">
              <svg className="w-full h-full" viewBox="0 0 100 30">
                <defs>
                  <linearGradient id={`sparkline-gradient-${title}`} x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="var(--color-electric-blue)" stopOpacity="0.3" />
                    <stop offset="100%" stopColor="var(--color-electric-blue)" stopOpacity="0" />
                  </linearGradient>
                </defs>
                <path
                  d={sparklinePath}
                  fill="none"
                  stroke="var(--color-electric-blue)"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d={`${sparklinePath} L 100,30 L 0,30 Z`}
                  fill={`url(#sparkline-gradient-${title})`}
                />
              </svg>
            </div>
          )}

          {/* Subtitle/Additional Info */}
          {subtitle && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-secondary font-mono">
                {subtitle}
              </span>
              <div className="flex gap-1">
                {[1,2,3].map((i) => (
                  <div
                    key={i}
                    className={`w-1 h-3 ${
                      i === 1 ? 'bg-electric-blue' : 'bg-light-slate'
                    } rounded-full`}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Corner Accent */}
          <div className="absolute top-0 right-0 w-16 h-16 overflow-hidden">
            <div className={`
              absolute -top-8 -right-8 w-16 h-16
              bg-gradient-to-br from-electric-blue/20 to-transparent
              transform rotate-45
            `} />
          </div>
        </div>
      </div>

      {/* Hover Glow Effect */}
      <div className={`
        absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100
        bg-gradient-to-br from-electric-blue/5 to-deep-blue/5
        blur-xl transition-opacity duration-500
        pointer-events-none
      `} />
    </motion.div>
  )
}