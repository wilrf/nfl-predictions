"use client"

import { motion, HTMLMotionProps } from "framer-motion"
import { LucideIcon } from "lucide-react"
import { forwardRef } from "react"
import { cn } from "@/lib/utils"

interface PalantirButtonProps extends Omit<HTMLMotionProps<"button">, "ref"> {
  variant?: "primary" | "secondary" | "ghost" | "danger" | "success"
  size?: "sm" | "md" | "lg"
  icon?: LucideIcon
  iconPosition?: "left" | "right"
  loading?: boolean
  glow?: boolean
}

export const PalantirButton = forwardRef<HTMLButtonElement, PalantirButtonProps>(({
  children,
  variant = "primary",
  size = "md",
  icon: Icon,
  iconPosition = "left",
  loading = false,
  glow = false,
  className,
  disabled,
  ...props
}, ref) => {

  const variants = {
    primary: cn(
      "bg-gradient-to-r from-electric-blue to-deep-blue",
      "border border-electric-blue/30",
      "text-white font-medium",
      "shadow-glow",
      "hover:shadow-glow-intense hover:border-electric-blue/50",
      "active:scale-[0.98]"
    ),
    secondary: cn(
      "bg-gradient-to-r from-dark-slate to-charcoal",
      "border border-electric-blue/20",
      "text-text-primary",
      "hover:border-electric-blue/40 hover:shadow-glow",
      "active:scale-[0.98]"
    ),
    ghost: cn(
      "bg-transparent",
      "border border-text-tertiary/20",
      "text-text-secondary",
      "hover:bg-white/5 hover:border-electric-blue/30 hover:text-text-primary",
      "active:bg-white/10"
    ),
    danger: cn(
      "bg-gradient-to-r from-critical/20 to-critical/10",
      "border border-critical/30",
      "text-critical",
      "hover:shadow-[0_0_20px_rgba(255,51,102,0.3)] hover:border-critical/50",
      "active:scale-[0.98]"
    ),
    success: cn(
      "bg-gradient-to-r from-success/20 to-success/10",
      "border border-success/30",
      "text-success",
      "hover:shadow-[0_0_20px_rgba(0,255,136,0.3)] hover:border-success/50",
      "active:scale-[0.98]"
    )
  }

  const sizes = {
    sm: "px-3 py-1.5 text-xs",
    md: "px-4 py-2 text-sm",
    lg: "px-6 py-3 text-base"
  }

  const iconSizes = {
    sm: "w-3 h-3",
    md: "w-4 h-4",
    lg: "w-5 h-5"
  }

  return (
    <motion.button
      ref={ref}
      whileTap={{ scale: 0.98 }}
      className={cn(
        // Base styles
        "relative inline-flex items-center justify-center",
        "rounded-lg font-mono uppercase tracking-wider",
        "transition-all duration-200",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        "overflow-hidden group",

        // Variant styles
        variants[variant],

        // Size styles
        sizes[size],

        // Custom classes
        className
      )}
      disabled={disabled || loading}
      {...props}
    >
      {/* Background scan effect */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
      </div>

      {/* Glow effect */}
      {glow && (
        <div className="absolute inset-0 animate-pulse-glow rounded-lg" />
      )}

      {/* Content */}
      <span className="relative z-10 flex items-center gap-2">
        {/* Loading spinner */}
        {loading ? (
          <div className={cn("animate-spin rounded-full border-2 border-current border-t-transparent", iconSizes[size])} />
        ) : (
          <>
            {Icon && iconPosition === "left" && (
              <Icon className={iconSizes[size]} />
            )}
            {children}
            {Icon && iconPosition === "right" && (
              <Icon className={iconSizes[size]} />
            )}
          </>
        )}
      </span>

      {/* Corner accents for primary variant */}
      {variant === "primary" && (
        <>
          <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-cyan/50 rounded-tl-lg" />
          <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-cyan/50 rounded-tr-lg" />
          <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-cyan/50 rounded-bl-lg" />
          <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-cyan/50 rounded-br-lg" />
        </>
      )}
    </motion.button>
  )
})

PalantirButton.displayName = "PalantirButton"