"use client"

import { Card, CardContent, CardHeader, CardTitle } from "../ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { motion } from "framer-motion"

interface ConfidenceData {
  bucket: string
  min_confidence: number
  count: number
  accuracy: number
}

interface ConfidenceChartProps {
  data: ConfidenceData[]
}

export function ConfidenceChart({ data }: ConfidenceChartProps) {
  const bucketNames: Record<string, string> = {
    very_high: '>75%',
    high: '65-75%',
    medium: '55-65%',
    low: '50-55%'
  }

  const chartData = data.map(d => ({
    bucket: bucketNames[d.bucket] || d.bucket,
    accuracy: (d.accuracy * 100).toFixed(1),
    count: d.count
  }))

  // Gradient colors from white to gray
  const colors = ['#ffffff', '#e5e5e5', '#6b6b6b', '#2d2d2d']

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">🎯</span>
            Confidence vs Accuracy
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d2d2d" />
              <XAxis
                dataKey="bucket"
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
                formatter={(value: number | string) => [
                  `${value}%`,
                  'Accuracy'
                ]}
              />
              <Bar dataKey="accuracy" radius={[8, 8, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </motion.div>
  )
}
