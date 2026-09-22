import {
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import {
  CHART,
  legendStyle,
  tickStyle,
  tooltipContentStyle,
  tooltipItemStyle,
  tooltipLabelStyle,
} from './theme';
import type { ReferenceSpec } from './LineChartPanel';

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
  yReferences?: ReferenceSpec[];
  tooltipFormatter?: (value: number | string, name: string) => [string, string];
  showLegend?: boolean;
}

export function ScatterChartPanel<T extends object>({
  data,
  xKey,
  yKey,
  label,
  color = '#58a6ff',
  colorFor,
  height = 260,
  xLabel,
  yLabel,
  yReferences = [],
  tooltipFormatter,
  showLegend = false,
}: Props<T>) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 18, bottom: xLabel ? 26 : 10, left: yLabel ? 12 : 0 }}>
        <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey={xKey}
          name={xLabel ?? xKey}
          stroke={CHART.axis}
          tick={tickStyle}
          tickLine={{ stroke: CHART.axis }}
          label={
            xLabel
              ? { value: xLabel, position: 'insideBottom', offset: -14, fill: CHART.axisLabel, fontSize: CHART.axisLabelFontSize }
              : undefined
          }
        />
        <YAxis
          type="number"
          dataKey={yKey}
          name={yLabel ?? yKey}
          stroke={CHART.axis}
          tick={tickStyle}
          tickLine={{ stroke: CHART.axis }}
          width={62}
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
        <ZAxis range={[64, 64]} />
        <Tooltip
          contentStyle={tooltipContentStyle}
          itemStyle={tooltipItemStyle}
          labelStyle={tooltipLabelStyle}
          cursor={{ strokeDasharray: '3 3', stroke: CHART.cursor }}
          formatter={tooltipFormatter}
        />
        {showLegend ? <Legend wrapperStyle={legendStyle} /> : null}
        {yReferences.map((ref) => (
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
        <Scatter name={label} data={data} fill={color} isAnimationActive={false}>
          {colorFor ? data.map((row, i) => <Cell key={i} fill={colorFor(row, i)} />) : null}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
