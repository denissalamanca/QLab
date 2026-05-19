import { Activity } from "lucide-react";

import { EmergencyFlatten } from "@/components/EmergencyFlatten";
import { StrategyTable } from "@/components/StrategyTable";
import { ValidationCharts } from "@/components/ValidationCharts";
import { useStrategies } from "@/hooks/useControlPlane";

export default function App() {
  const { data: strategies = [], isLoading, error } = useStrategies();

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            <div>
              <h1 className="text-lg font-semibold leading-tight">AFML Quant Lab — Control Plane</h1>
              <p className="text-xs text-muted-foreground">CEO Human-in-the-Loop governance</p>
            </div>
          </div>
          <EmergencyFlatten />
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-6 py-6">
        <ValidationCharts strategies={strategies} />
        <StrategyTable strategies={strategies} isLoading={isLoading} error={error} />
      </main>
    </div>
  );
}
