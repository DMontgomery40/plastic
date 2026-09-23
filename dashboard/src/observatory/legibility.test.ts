// The dashboard legibility floor, checked on the observatory sources: no text below 11px, colors only from tokens,
// no opacity-based text de-emphasis, no emoji.
import { describe, expect, it } from 'vitest';
import TAILWIND from '../../tailwind.config.js?raw';

const RAW = import.meta.glob(['./*.ts', './*.tsx', '!./*.test.ts', '!./*.test.tsx', '!./testData.ts'], { eager: true, query: '?raw', import: 'default' }) as Record<string, string>;
const sources = Object.entries(RAW);

describe('observatory legibility floor', () => {
  it('has sources to check', () => expect(sources.length).toBeGreaterThan(3));

  for (const [file, text] of sources) {
    it(`${file}: colors come from tokens, not literals`, () => {
      expect(text.match(/#[0-9a-fA-F]{3,8}\b/g) ?? []).toEqual([]);
      expect(text.match(/\brgba?\(/g) ?? []).toEqual([]);
    });

    it(`${file}: nothing below 11px`, () => {
      const px = [...text.matchAll(/text-\[(\d+(?:\.\d+)?)px\]|fontSize[=:]\s*\{?\s*(\d+(?:\.\d+)?)/g)].map((m) => Number(m[1] ?? m[2]));
      expect(px.filter((v) => v < 11)).toEqual([]);
    });

    it(`${file}: no opacity on text, no emoji`, () => {
      expect(text.match(/\b(?:text-opacity|opacity)-\d+/g) ?? []).toEqual([]);
      expect(text.match(/\p{Extended_Pictographic}/gu) ?? []).toEqual([]);
    });
  }
});

// The Learning view is newer and held to the full floor from ~/.claude/rules/design-legibility.md: labels >= 11.5px,
// body text >= 14px, body ink >= 7:1 and every text token >= 4.5:1 against the surfaces it sits on, computed from the
// Tailwind tokens themselves.

const LEARNING_SOURCES = Object.entries(RAW).filter(([file]) => /Learning|learning/.test(file));

function tokens(group: string): Record<string, string> {
  const body = TAILWIND.match(new RegExp(`\\b${group}:\\s*\\{([^}]*)\\}`))?.[1] ?? '';
  return Object.fromEntries([...body.matchAll(/(\w+):\s*'(#[0-9a-fA-F]{6})'/g)].map((m) => [m[1], m[2]]));
}

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

describe('observatory legibility floor: font smoothing', () => {
  for (const [file, text] of sources) {
    it(`${file}: no antialiased smoothing outside a 2dppx media query`, () => {
      expect(text.match(/\bantialiased\b|font-smoothing/g) ?? []).toEqual([]);
    });
  }
});

describe('Learning view legibility floor', () => {
  it('has Learning sources to check', () => expect(LEARNING_SOURCES.map(([f]) => f).sort()).toEqual(['./LearningView.tsx', './learning.ts']));

  for (const [file, text] of LEARNING_SOURCES) {
    it(`${file}: labels >= 11.5px, paragraphs and tables >= 14px`, () => {
      expect(text.match(/\btext-micro\b|text-\[\d/g) ?? []).toEqual([]);
      for (const m of text.matchAll(/<(p|table)\b[^>]*className="([^"]*)"/g)) expect(m[2]).toMatch(/\btext-(?:base|md|lg|xl|2xl|3xl)\b/);
    });
  }

  it('every text token it uses clears 4.5:1, and body ink clears 7:1, on each surface it uses', () => {
    const ink = tokens('ink');
    const colors: Record<string, string> = {
      'ink-primary': ink.primary, 'ink-secondary': ink.secondary, 'ink-muted': ink.muted,
      ...Object.fromEntries(Object.entries(tokens('status')).map(([k, v]) => [`status-${k}`, v])),
      accent: tokens('accent').DEFAULT,
    };
    const surface = tokens('surface');
    const grounds = { page: surface.DEFAULT, raised: surface.raised, overlay: surface.overlay, 'accent-soft': tokens('accent').soft };
    const used = new Set(LEARNING_SOURCES.flatMap(([, text]) => [...text.matchAll(/\btext-(ink-(?:primary|secondary|muted)|status-\w+|accent)\b/g)].map((m) => m[1])));
    expect(used.size).toBeGreaterThan(3);
    for (const name of used) {
      expect(colors[name], name).toMatch(/^#/);
      for (const [ground, hex] of Object.entries(grounds)) {
        if (name === 'accent' ? ground !== 'accent-soft' && ground !== 'raised' : ground === 'accent-soft') continue;
        const floor = name === 'ink-primary' || name === 'ink-secondary' ? 7 : 4.5;
        expect([name, ground, contrast(colors[name], hex) >= floor]).toEqual([name, ground, true]);
      }
    }
  });
});
