import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  CHART,
  legendStyle,
  tickStyle,
  tooltipContentStyle,
  tooltipItemStyle,
  tooltipLabelStyle,
  type SeriesSpec,
} from './theme';

export interface ReferenceSpec {
  value: number;
  label: string;
  color?: string;
}

interface Props<T> {
  data: T[];
  xKey: string;
  series: SeriesSpec[];
  height?: number;
  logScale?: boolean;
  xLabel?: string;
  yLabel?: string;
  references?: ReferenceSpec[];
  showLegend?: boolean;
  dots?: boolean;
  onPointClick?: (index: number) => void;
  /** Required. The chart's programmatic label; charts are images to assistive tech. */
  ariaLabel: string;
}

export function LineChartPanel<T extends object>({
  data,
  xKey,
  series,
  height = 220,
  logScale = false,
  xLabel,
  yLabel,
  references = [],
  showLegend = true,
  ariaLabel,
  dots = false,
  onPointClick,
}: Props<T>) {
  const handleClick = onPointClick
    ? (state: { activeTooltipIndex?: number }) => {
        if (typeof state?.activeTooltipIndex === 'number') onPointClick(state.activeTooltipIndex);
      }
    : undefined;

  return (
    <figure role="img" aria-label={ariaLabel} className="m-0">
      <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 14, bottom: xLabel ? 22 : 6, left: yLabel ? 10 : 0 }} onClick={handleClick}>
        <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey={xKey}
          stroke={CHART.axis}
          tick={tickStyle}
          tickLine={{ stroke: CHART.axis }}
          interval="preserveStartEnd"
          minTickGap={28}
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
          scale={logScale ? 'log' : 'auto'}
          domain={logScale ? ['auto', 'auto'] : undefined}
          allowDataOverflow={false}
          width={60}
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
          cursor={{ stroke: CHART.cursor, strokeWidth: 1 }}
        />
        {showLegend && series.length > 1 ? (
          <Legend verticalAlign="top" align="right" height={20} wrapperStyle={legendStyle} />
        ) : null}
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
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dashed ? '6 4' : undefined}
            dot={dots ? { r: 2.5, fill: s.color, strokeWidth: 0 } : false}
            activeDot={{ r: 4, strokeWidth: 0 }}
            isAnimationActive={false}
            connectNulls
          />
        ))}
      </LineChart>
      </ResponsiveContainer>
    </figure>
  );
}
