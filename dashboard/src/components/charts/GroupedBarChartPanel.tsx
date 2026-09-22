import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import {
  CHART,
  legendStyle,
  tickStyle,
  tooltipContentStyle,
  tooltipItemStyle,
  tooltipLabelStyle,
  type SeriesSpec,
} from './theme';
import type { ReferenceSpec } from './LineChartPanel';

interface Props<T> {
  data: T[];
  xKey: string;
  series: SeriesSpec[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  references?: ReferenceSpec[];
  ariaLabel: string;
}

/**
 * Bars grouped side by side, one group per category. Used where the comparison
 * between the series is the point, not their sum: the gap between what an
 * attack achieved against the harness and what it achieved without it.
 */
export function GroupedBarChartPanel<T extends object>({
  data,
  xKey,
  series,
  height = 240,
  xLabel,
  yLabel,
  references = [],
  ariaLabel,
}: Props<T>) {
  return (
    <figure role="img" aria-label={ariaLabel} className="m-0">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 24, right: 16, bottom: xLabel ? 22 : 6, left: yLabel ? 10 : 0 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey={xKey}
            stroke={CHART.axis}
            tick={tickStyle}
            tickLine={{ stroke: CHART.axis }}
            interval={0}
            label={
              xLabel
                ? { value: xLabel, position: 'insideBottom', offset: -12, fill: CHART.axisLabel, fontSize: CHART.axisLabelFontSize }
                : undefined
            }
          />
          <YAxis
            stroke={CHART.axis}
            tick={tickStyle}
            tickLine={{ stroke: CHART.axis }}
            width={68}
            label={
              yLabel
                ? {
                    value: yLabel,
                    angle: -90,
                    position: 'insideLeft',
                    fill: CHART.axisLabel,
                    fontSize: CHART.axisLabelFontSize,
                    style: { textAnchor: 'middle' },
                  }
                : undefined
            }
          />
          <Tooltip
            contentStyle={tooltipContentStyle}
            itemStyle={tooltipItemStyle}
            labelStyle={tooltipLabelStyle}
            cursor={{ fill: '#1c242e' }}
          />
          <Legend verticalAlign="top" align="right" height={20} wrapperStyle={legendStyle} />
          <ReferenceLine y={0} stroke={CHART.axis} strokeWidth={1} />
          {references.map((ref) => (
            <ReferenceLine
              key={ref.label}
              y={ref.value}
              stroke={ref.color ?? CHART.reference}
              strokeDasharray="5 4"
              strokeWidth={1.5}
              label={{
                value: ref.label,
                position: 'right',
                fill: ref.color ?? CHART.reference,
                fontSize: CHART.axisLabelFontSize,
              }}
            />
          ))}
          {series.map((s) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </figure>
  );
}
