"use client"

import { motion } from "framer-motion"
import { ReactNode } from "react"
import { cn } from "@/lib/utils"
import { X, Maximize2, Minimize2, MoreVertical } from "lucide-react"
import { useState } from "react"

interface PalantirPanelProps {
  title: string
  subtitle?: string
  children: ReactNode
  className?: string
  collapsible?: boolean
  closable?: boolean
  onClose?: () => void
  toolbar?: ReactNode
  status?: "idle" | "loading" | "live" | "error"
}

export function PalantirPanel({
  title,
  subtitle,
  children,
  className,
  collapsible = false,
  closable = false,
  onClose,
  toolbar,
  status = "idle"
}: PalantirPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [isMaximized, setIsMaximized] = useState(false)

  const statusIndicators = {
    idle: null,
    loading: (
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 bg-electric-blue rounded-full animate-pulse" />
        <span className="text-micro text-text-tertiary">LOADING</span>
      </div>
    ),
    live: (
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
        <span className="text-micro text-success">LIVE</span>
      </div>
    ),
    error: (
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 bg-critical rounded-full" />
        <span className="text-micro text-critical">ERROR</span>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{
        opacity: 1,
        scale: 1,
        height: isExpanded ? "auto" : "auto"
      }}
      transition={{
        duration: 0.3,
        ease: [0.16, 1, 0.3, 1]
      }}
      className={cn(
        "relative rounded-lg overflow-hidden",
        "bg-gradient-to-br from-dark-slate/95 to-charcoal/95",
        "backdrop-blur-xl",
        "border border-electric-blue/10",
        "shadow-panel",
        isMaximized && "fixed inset-4 z-50",
        className
      )}
    >
      {/* Grid Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0 bg-grid-pattern bg-grid" />
      </div>

      {/* Header */}
      <div className="relative z-10 border-b border-electric-blue/10 bg-charcoal/50">
        <div className="flex items-center justify-between px-4 py-3">
          {/* Title Section */}
          <div className="flex items-center gap-4">
            {/* Drag Handle */}
            <div className="flex flex-col gap-0.5 opacity-30 hover:opacity-60 cursor-move transition-opacity">
              <div className="w-4 h-0.5 bg-text-secondary rounded-full" />
              <div className="w-4 h-0.5 bg-text-secondary rounded-full" />
              <div className="w-4 h-0.5 bg-text-secondary rounded-full" />
            </div>

            <div>
              <h3 className="text-sm font-medium text-text-primary font-display tracking-wide">
                {title}
              </h3>
              {subtitle && (
                <p className="text-micro text-text-tertiary mt-0.5">
                  {subtitle}
                </p>
              )}
            </div>

            {statusIndicators[status]}
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            {/* Custom Toolbar */}
            {toolbar}

            {/* Panel Controls */}
            <div className="flex items-center gap-1">
              {collapsible && (
                <button
                  onClick={() => setIsMaximized(!isMaximized)}
                  className={cn(
                    "p-1.5 rounded-md",
                    "text-text-tertiary hover:text-text-primary",
                    "hover:bg-white/5 transition-all"
                  )}
                >
                  {isMaximized ? (
                    <Minimize2 className="w-4 h-4" />
                  ) : (
                    <Maximize2 className="w-4 h-4" />
                  )}
                </button>
              )}

              <button
                className={cn(
                  "p-1.5 rounded-md",
                  "text-text-tertiary hover:text-text-primary",
                  "hover:bg-white/5 transition-all"
                )}
              >
                <MoreVertical className="w-4 h-4" />
              </button>

              {closable && (
                <button
                  onClick={onClose}
                  className={cn(
                    "p-1.5 rounded-md",
                    "text-text-tertiary hover:text-critical",
                    "hover:bg-critical/10 transition-all"
                  )}
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Loading Bar */}
        {status === "loading" && (
          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-charcoal overflow-hidden">
            <div className="h-full bg-gradient-to-r from-transparent via-electric-blue to-transparent animate-scan-line" />
          </div>
        )}
      </div>

      {/* Content */}
      <motion.div
        animate={{
          opacity: isExpanded ? 1 : 0,
          height: isExpanded ? "auto" : 0
        }}
        transition={{
          duration: 0.3,
          ease: [0.16, 1, 0.3, 1]
        }}
        className="relative z-10 overflow-hidden"
      >
        <div className="p-4">
          {children}
        </div>
      </motion.div>

      {/* Corner Accents */}
      <div className="absolute top-0 left-0 w-8 h-8 overflow-hidden pointer-events-none">
        <div className="absolute -top-4 -left-4 w-8 h-8 border-t border-l border-electric-blue/20" />
      </div>
      <div className="absolute top-0 right-0 w-8 h-8 overflow-hidden pointer-events-none">
        <div className="absolute -top-4 -right-4 w-8 h-8 border-t border-r border-electric-blue/20" />
      </div>
      <div className="absolute bottom-0 left-0 w-8 h-8 overflow-hidden pointer-events-none">
        <div className="absolute -bottom-4 -left-4 w-8 h-8 border-b border-l border-electric-blue/20" />
      </div>
      <div className="absolute bottom-0 right-0 w-8 h-8 overflow-hidden pointer-events-none">
        <div className="absolute -bottom-4 -right-4 w-8 h-8 border-b border-r border-electric-blue/20" />
      </div>
    </motion.div>
  )
}