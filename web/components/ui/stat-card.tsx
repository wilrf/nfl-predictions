"use client"

import { Card } from "./card"
import { motion } from "framer-motion"
import { LucideIcon } from "lucide-react"

interface StatCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon?: LucideIcon
  trend?: "up" | "down" | "neutral"
  delay?: number
}

export function StatCard({ title, value, subtitle, icon: Icon, trend, delay = 0 }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -8, scale: 1.02 }}
      className="group"
    >
      <Card className="relative overflow-hidden cursor-pointer border border-white/5 shadow-2xl hover:shadow-[0_20px_60px_-15px_rgba(255,255,255,0.15)] transition-all duration-500">
        {/* Animated gradient background */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/[0.08] via-white/[0.02] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

        {/* Subtle glow effect on hover */}
        <div className="absolute -inset-[1px] bg-gradient-to-br from-white/20 via-transparent to-transparent opacity-0 group-hover:opacity-100 blur-sm transition-opacity duration-500 -z-10" />

        {/* Accent line at top */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-white/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

        <div className="relative z-10">
          <div className="flex items-start justify-between mb-4">
            <div className="text-medium-gray text-sm font-medium uppercase tracking-wider group-hover:text-white/80 transition-colors duration-300">
              {title}
            </div>
            {Icon && (
              <div className="p-2.5 rounded-xl bg-white/5 group-hover:bg-white/10 transition-all duration-300 group-hover:scale-110 group-hover:rotate-6">
                <Icon className="w-5 h-5 text-white/80 group-hover:text-white transition-colors duration-300" />
              </div>
            )}
          </div>

          <div className="mb-2">
            <div className="text-5xl font-bold text-white tracking-tight group-hover:scale-105 transition-transform duration-300 origin-left">
              {value}
            </div>
          </div>

          {subtitle && (
            <div className="text-sm text-medium-gray group-hover:text-white/70 transition-colors duration-300">
              {subtitle}
            </div>
          )}

          {trend && (
            <div className={`mt-3 inline-flex items-center gap-1 text-xs font-semibold px-2 py-1 rounded-full ${
              trend === "up" ? "bg-green-500/10 text-green-400" :
              trend === "down" ? "bg-red-500/10 text-red-400" :
              "bg-gray-500/10 text-gray-400"
            } transition-all duration-300 group-hover:scale-110`}>
              <span>{trend === "up" ? "↑" : trend === "down" ? "↓" : "→"}</span>
              <span className="uppercase tracking-wider">{trend}</span>
            </div>
          )}
        </div>
      </Card>
    </motion.div>
  )
}
