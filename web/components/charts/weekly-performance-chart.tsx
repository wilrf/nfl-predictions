"use client"

import { Card, CardContent, CardHeader, CardTitle } from "../ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { motion } from "framer-motion"

interface WeeklyData {
  week: number
  spread_accuracy: number
  total_accuracy: number
  games: number
}

interface WeeklyPerformanceChartProps {
  data: WeeklyData[]
}

export function WeeklyPerformanceChart({ data }: WeeklyPerformanceChartProps) {
  const chartData = data.map(d => ({
    week: `Week ${d.week}`,
    Spread: (d.spread_accuracy * 100).toFixed(1),
    Total: (d.total_accuracy * 100).toFixed(1),
  }))

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">📊</span>
            Weekly Performance Trends
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d2d2d" />
              <XAxis
                dataKey="week"
                stroke="#6b6b6b"
                tick={{ fill: '#e5e5e5' }}
                style={{ fontSize: '12px' }}
              />
              <YAxis
                stroke="#6b6b6b"
                tick={{ fill: '#e5e5e5' }}
                style={{ fontSize: '12px' }}
                domain={[0, 100]}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #2d2d2d',
                  borderRadius: '8px',
                  color: '#fff'
                }}
                labelStyle={{ color: '#e5e5e5' }}
              />
              <Legend
                wrapperStyle={{ color: '#e5e5e5' }}
                iconType="line"
              />
              <Line
                type="monotone"
                dataKey="Spread"
                stroke="#ffffff"
                strokeWidth={3}
                dot={{ fill: '#ffffff', r: 5 }}
                activeDot={{ r: 8, fill: '#ffffff' }}
              />
              <Line
                type="monotone"
                dataKey="Total"
                stroke="#6b6b6b"
                strokeWidth={3}
                dot={{ fill: '#6b6b6b', r: 5 }}
                activeDot={{ r: 8, fill: '#6b6b6b' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </motion.div>
  )
}
