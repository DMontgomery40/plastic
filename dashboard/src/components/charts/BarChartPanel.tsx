import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { CHART, tickStyle, tooltipContentStyle, tooltipItemStyle, tooltipLabelStyle } from './theme';

interface Props<T> {
  data: T[];
  xKey: string;
  yKey: string;
  label: string;
  color?: string;
  colorFor?: (row: T, index: number) => string;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  /** Required. The chart's programmatic label; charts are images to assistive tech. */
  ariaLabel: string;
}

export function BarChartPanel<T extends object>({
  data,
  xKey,
  yKey,
  label,
  color = '#58a6ff',
  colorFor,
  height = 200,
  xLabel,
  yLabel,
  ariaLabel,
}: Props<T>) {
  return (
    <figure role="img" aria-label={ariaLabel} className="m-0">
      <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 14, bottom: xLabel ? 22 : 6, left: yLabel ? 10 : 0 }}>
        <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey={xKey}
          stroke={CHART.axis}
          tick={tickStyle}
          tickLine={{ stroke: CHART.axis }}
          interval="preserveStartEnd"
          minTickGap={22}
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
          width={58}
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
        <Bar dataKey={yKey} name={label} fill={color} isAnimationActive={false}>
          {colorFor
            ? data.map((row, i) => <Cell key={i} fill={colorFor(row, i)} />)
            : null}
        </Bar>
      </BarChart>
      </ResponsiveContainer>
    </figure>
  );
}
