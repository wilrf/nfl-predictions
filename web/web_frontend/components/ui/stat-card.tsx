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
      whileHover={{ y: -5, scale: 1.02 }}
    >
      <Card className="relative overflow-hidden group cursor-pointer">
        {/* Background gradient on hover */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

        <div className="relative z-10">
          <div className="flex items-start justify-between mb-4">
            <div className="text-medium-gray text-sm font-medium uppercase tracking-wider">
              {title}
            </div>
            {Icon && (
              <div className="p-2 rounded-lg bg-white/5">
                <Icon className="w-5 h-5 text-white" />
              </div>
            )}
          </div>

          <div className="mb-2">
            <div className="text-5xl font-bold text-white tracking-tight">
              {value}
            </div>
          </div>

          {subtitle && (
            <div className="text-sm text-medium-gray">
              {subtitle}
            </div>
          )}

          {trend && (
            <div className={`mt-3 text-xs font-medium ${
              trend === "up" ? "text-light-gray" :
              trend === "down" ? "text-medium-gray" :
              "text-medium-gray"
            }`}>
              {trend === "up" ? "↑" : trend === "down" ? "↓" : "→"}
            </div>
          )}
        </div>
      </Card>
    </motion.div>
  )
}
