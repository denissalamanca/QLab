import { AlertTriangle } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useFlatten } from "@/hooks/useControlPlane";
import { flattenMessage } from "@/lib/api";

/**
 * Emergency flatten kill-switch (Blueprint §11.1).
 *
 * Signature-only (no TOTP) so the kill-switch is never blocked by a code
 * window. A fresh nonce is minted per attempt — the CEO signs
 * `afml:flatten:<nonce>` and pastes the signature.
 */
export function EmergencyFlatten() {
  const [open, setOpen] = useState(false);
  // Fresh nonce + timestamp per open — layered anti-replay (audit V1).
  const nonce = useMemo(() => crypto.randomUUID(), [open]);
  const timestampMs = useMemo(() => Date.now(), [open]);
  const [reason, setReason] = useState("");
  const [signedToken, setSignedToken] = useState("");
  const flatten = useFlatten();

  const handleSubmit = () => {
    flatten.mutate(
      {
        nonce,
        timestamp_ms: timestampMs,
        signed_token: signedToken.trim(),
        reason: reason.trim() || "manual kill-switch",
      },
      {
        onSuccess: () => {
          setOpen(false);
          setSignedToken("");
          setReason("");
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="destructive" data-testid="flatten-open">
          <AlertTriangle className="h-4 w-4" /> Emergency Flatten
        </Button>
      </DialogTrigger>
      <DialogContent data-testid="flatten-modal">
        <DialogHeader>
          <DialogTitle>Liquidate all positions</DialogTitle>
          <DialogDescription>
            Closes every open position immediately. Requires a CEO signature over the nonce below.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div>
            <label className="text-xs font-medium text-muted-foreground">Message to sign</label>
            <code className="mt-1 block break-all rounded-md bg-secondary px-3 py-2 text-xs" data-testid="flatten-message">
              {flattenMessage(nonce, timestampMs)}
            </code>
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground" htmlFor="reason">
              Reason
            </label>
            <Input
              id="reason"
              data-testid="flatten-reason"
              placeholder="e.g. daily drawdown breach"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
            />
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground" htmlFor="flatten-sig">
              Ed25519 signature (hex)
            </label>
            <Input
              id="flatten-sig"
              data-testid="flatten-signature"
              autoComplete="off"
              placeholder="128 hex chars"
              value={signedToken}
              onChange={(e) => setSignedToken(e.target.value)}
            />
          </div>
          {flatten.isError && (
            <p className="text-sm text-destructive" data-testid="flatten-error" role="alert">
              {flatten.error.message}
            </p>
          )}
          {flatten.isSuccess && (
            <p className="text-sm text-primary" data-testid="flatten-success">
              Flattened {flatten.data.n_positions_closed} position(s).
            </p>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="destructive"
            onClick={handleSubmit}
            disabled={flatten.isPending || signedToken.trim() === ""}
            data-testid="flatten-submit"
          >
            {flatten.isPending ? "Liquidating…" : "Confirm Flatten"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
