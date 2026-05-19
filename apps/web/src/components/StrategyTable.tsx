import { ApprovalModal } from "@/components/ApprovalModal";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Strategy } from "@/lib/api";
import { formatNumber } from "@/lib/utils";

interface Props {
  strategies: Strategy[];
  isLoading: boolean;
  error: Error | null;
}

const PBO_GATE = 0.05;
const DSR_GATE = 1.0;

export function StrategyTable({ strategies, isLoading, error }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Strategies awaiting CEO sign-off</CardTitle>
        <CardDescription>CPCV-validated strategies eligible for Paper → Live promotion.</CardDescription>
      </CardHeader>
      <CardContent>
        {error && (
          <p className="text-sm text-destructive" data-testid="strategies-error">
            {error.message}
          </p>
        )}
        {isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
        {!isLoading && !error && strategies.length === 0 && (
          <p className="text-sm text-muted-foreground" data-testid="strategies-empty">
            No strategies awaiting sign-off.
          </p>
        )}
        {strategies.length > 0 && (
          <Table data-testid="strategies-table">
            <TableHeader>
              <TableRow>
                <TableHead>Asset</TableHead>
                <TableHead>Family</TableHead>
                <TableHead>Events</TableHead>
                <TableHead>PBO</TableHead>
                <TableHead>DSR</TableHead>
                <TableHead>Brain-1 recall</TableHead>
                <TableHead className="text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {strategies.map((s) => (
                <TableRow key={s.experiment_id} data-testid={`strategy-row-${s.experiment_id}`}>
                  <TableCell className="font-medium">{s.asset}</TableCell>
                  <TableCell>{s.algorithmic_family}</TableCell>
                  <TableCell>{s.num_events_triggered}</TableCell>
                  <TableCell>
                    <Badge variant={(s.pbo ?? 1) < PBO_GATE ? "default" : "destructive"}>
                      {formatNumber(s.pbo)}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={(s.dsr ?? 0) > DSR_GATE ? "default" : "destructive"}>
                      {formatNumber(s.dsr, 2)}
                    </Badge>
                  </TableCell>
                  <TableCell>{formatNumber(s.brain_1_recall, 2)}</TableCell>
                  <TableCell className="text-right">
                    <ApprovalModal strategy={s} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
