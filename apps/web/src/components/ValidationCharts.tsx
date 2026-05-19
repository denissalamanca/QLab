import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { Strategy } from "@/lib/api";

// Blueprint ship gates: PBO must be < 0.05, DSR must be > 1.0.
const PBO_GATE = 0.05;
const DSR_GATE = 1.0;

interface Props {
  strategies: Strategy[];
}

export function ValidationCharts({ strategies }: Props) {
  const data = strategies.map((s) => ({
    label: `${s.asset}/${s.algorithmic_family}`,
    pbo: s.pbo ?? 0,
    dsr: s.dsr ?? 0,
  }));

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Probability of Backtest Overfitting</CardTitle>
          <CardDescription>Ship gate: PBO &lt; {PBO_GATE.toFixed(2)} (green line)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 24%)" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} stroke="hsl(215 20% 65%)" />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="hsl(215 20% 65%)" />
              <Tooltip
                contentStyle={{ background: "hsl(222 44% 14%)", border: "1px solid hsl(217 33% 24%)" }}
              />
              <ReferenceLine y={PBO_GATE} stroke="hsl(142 71% 45%)" strokeDasharray="4 4" />
              <Bar dataKey="pbo" radius={[4, 4, 0, 0]}>
                {data.map((d, i) => (
                  <Cell
                    key={i}
                    fill={d.pbo < PBO_GATE ? "hsl(142 71% 45%)" : "hsl(0 72% 51%)"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Deflated Sharpe Ratio</CardTitle>
          <CardDescription>Ship gate: DSR &gt; {DSR_GATE.toFixed(1)} (green line)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 24%)" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} stroke="hsl(215 20% 65%)" />
              <YAxis tick={{ fontSize: 11 }} stroke="hsl(215 20% 65%)" />
              <Tooltip
                contentStyle={{ background: "hsl(222 44% 14%)", border: "1px solid hsl(217 33% 24%)" }}
              />
              <ReferenceLine y={DSR_GATE} stroke="hsl(142 71% 45%)" strokeDasharray="4 4" />
              <Bar dataKey="dsr" radius={[4, 4, 0, 0]}>
                {data.map((d, i) => (
                  <Cell
                    key={i}
                    fill={d.dsr > DSR_GATE ? "hsl(142 71% 45%)" : "hsl(0 72% 51%)"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
