// The dashboard legibility floor, checked on the observatory sources: no text below 11px, colors only from tokens,
// no opacity-based text de-emphasis, no emoji.
import { describe, expect, it } from 'vitest';

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
