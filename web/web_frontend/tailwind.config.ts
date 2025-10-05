import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Core Background Layers
        'deep-black': '#000000',
        'void': '#0A0A0B',
        'charcoal': '#0F1114',
        'dark-slate': '#1A1D23',
        'slate': '#1E2028',
        'medium-slate': '#2A2D36',
        'light-slate': '#3A3D47',

        // Primary Accent Colors
        'electric-blue': '#00D4FF',
        'cyan': '#00F0FF',
        'deep-blue': '#0066CC',

        // Status Colors
        'warning': '#FFB800',
        'critical': '#FF3366',
        'success': '#00FF88',

        // Text Colors
        'text-primary': '#FFFFFF',
        'text-secondary': '#B8BCC8',
        'text-tertiary': '#6B7280',
        'text-disabled': '#3A3D47',

        // Data Visualization Palette
        'data': {
          '1': '#00D4FF',
          '2': '#00FF88',
          '3': '#FFB800',
          '4': '#FF3366',
          '5': '#A78BFA',
          '6': '#F472B6',
        }
      },
      fontFamily: {
        'sans': ['Inter', '-apple-system', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
        'display': ['Space Grotesk', 'Inter', 'sans-serif'],
      },
      fontSize: {
        'micro': '0.625rem',  // 10px
        'xs': '0.75rem',      // 12px
        'sm': '0.875rem',     // 14px
        'base': '1rem',       // 16px
        'lg': '1.25rem',      // 20px
        'xl': '1.5rem',       // 24px
        '2xl': '1.75rem',     // 28px
        '3xl': '2.25rem',     // 36px
        '4xl': '3rem',        // 48px
        '5xl': '4.5rem',      // 72px
      },
      spacing: {
        'micro': '0.25rem',   // 4px
        'xs': '0.5rem',       // 8px
        'sm': '1rem',         // 16px
        'md': '1.5rem',       // 24px
        'lg': '2rem',         // 32px
        'xl': '3rem',         // 48px
        '2xl': '4rem',        // 64px
      },
      boxShadow: {
        'glow': '0 0 20px rgba(0, 212, 255, 0.3)',
        'glow-intense': '0 0 40px rgba(0, 212, 255, 0.6)',
        'inner-dark': 'inset 0 0 20px rgba(0, 0, 0, 0.5)',
        'panel': '0 0 20px rgba(0, 212, 255, 0.05), inset 0 0 20px rgba(0, 0, 0, 0.5)',
        'neumorphic': '5px 5px 10px rgba(0, 0, 0, 0.5), -5px -5px 10px rgba(42, 45, 54, 0.3)',
        'neumorphic-pressed': 'inset 5px 5px 10px rgba(0, 0, 0, 0.5), inset -5px -5px 10px rgba(42, 45, 54, 0.3)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'glass-gradient': 'linear-gradient(145deg, rgba(26, 29, 35, 0.8), rgba(15, 17, 20, 0.8))',
        'panel-gradient': 'linear-gradient(145deg, #1A1D23, #0F1114)',
        'grid-pattern': `linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
                         linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px)`,
        'holographic': `linear-gradient(135deg,
                         rgba(0, 212, 255, 0.1),
                         rgba(0, 240, 255, 0.05),
                         rgba(0, 102, 204, 0.1))`,
      },
      backgroundSize: {
        'grid': '50px 50px',
      },
      animation: {
        'fade-in': 'fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'slide-in': 'slideIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.87, 0, 0.13, 1) infinite',
        'data-stream': 'data-stream 3s cubic-bezier(0.87, 0, 0.13, 1) infinite',
        'scan-line': 'scan-line 3s linear infinite',
        'holographic-shift': 'holographic-shift 3s ease-in-out infinite',
        'glitch': 'glitch 2s infinite',
      },
      backdropBlur: {
        'xs': '2px',
        'xl': '20px',
        '2xl': '40px',
      },
      transitionTimingFunction: {
        'expo-out': 'cubic-bezier(0.16, 1, 0.3, 1)',
        'expo-in-out': 'cubic-bezier(0.87, 0, 0.13, 1)',
      },
    },
  },
  plugins: [],
}

export default config