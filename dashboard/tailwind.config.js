/** @type {import('tailwindcss').Config} */
//
// Token rules (see ~/.claude/rules/design-legibility.md):
//   - Every color used anywhere in src/ is defined here. No ad-hoc hex in components.
//   - Contrast is measured against surface.DEFAULT (#0d1117), the page ground.
//     text.primary 17.5:1, text.secondary 10.7:1, text.muted 7.4:1 (all clear the 7:1
//     body floor, so the muted tier is still safe for real text and nothing needs
//     opacity to de-emphasize).
//   - status.* are the bright, readable tier used for text and chart strokes;
//     statusFill.* are the deeper companions used only for filled marks and bars.
//   - Smallest defined type is 11px (`text-micro`); body is 14px (`text-base`).
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#0d1117', // page ground
          raised: '#151b23', // cards and panels
          overlay: '#1c242e', // inputs, hovered rows, chips
          inset: '#090d12', // code blocks, wells
        },
        edge: {
          DEFAULT: '#2c3642', // default 1px borders and chart grid
          strong: '#3d4a59', // emphasized dividers, focus rings
        },
        ink: {
          primary: '#e9eff5', // body text, 17.5:1
          secondary: '#b8c4d0', // supporting text, 10.7:1
          muted: '#94a3b4', // labels and axis ticks, 7.4:1
          inverse: '#080c11', // text on a saturated fill
        },
        accent: {
          DEFAULT: '#58a6ff', // interactive: links, selection, focus
          hover: '#79b8ff',
          // Selected-row wash. Solid, and dark enough that text keeps its
          // contrast against the COMPOSITED background: accent 7.1:1,
          // ink-secondary 10.2:1, ink-primary 16.7:1, status-commit 9.1:1.
          soft: '#0c1726',
        },
        status: {
          commit: '#3fd17a',
          rollback: '#ff6b6b',
          scale: '#f0b429',
          project: '#58a6ff',
          readonly: '#94a3b4',
          running: '#56d4dd',
          failed: '#ff6b6b',
        },
        statusFill: {
          commit: '#1f7a45',
          rollback: '#9c2b2b',
          scale: '#8a6410',
          project: '#1f4f8f',
          readonly: '#48545f',
        },
        series: {
          a: '#58a6ff', // base / first series
          b: '#3fd17a', // frozen / second series
          c: '#f0b429', // adaptive / third series
          d: '#c792ea', // fourth series
          e: '#56d4dd', // fifth series
        },
      },
      fontFamily: {
        sans: ['Archivo', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      fontSize: {
        // nothing below 11px
        micro: ['11px', { lineHeight: '15px' }],
        label: ['11.5px', { lineHeight: '16px' }],
        xs: ['12px', { lineHeight: '17px' }],
        sm: ['13px', { lineHeight: '19px' }],
        base: ['14px', { lineHeight: '21px' }],
        md: ['15px', { lineHeight: '23px' }],
        lg: ['17px', { lineHeight: '25px' }],
        xl: ['21px', { lineHeight: '28px' }],
        '2xl': ['27px', { lineHeight: '34px' }],
        '3xl': ['36px', { lineHeight: '42px' }],
      },
      borderRadius: {
        DEFAULT: '6px',
        lg: '10px',
      },
    },
  },
  plugins: [],
};
