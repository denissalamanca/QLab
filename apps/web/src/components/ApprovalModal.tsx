import { ShieldCheck } from "lucide-react";
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
import { useApprove } from "@/hooks/useControlPlane";
import { approvalMessage, type Strategy } from "@/lib/api";

interface Props {
  strategy: Strategy;
}

/**
 * CEO 2FA / cryptographic approval modal (Blueprint §11.2).
 *
 * The CEO private key never enters the browser. The CEO signs the canonical
 * message — `afml:approve:<experiment_id>` — on their own device (CLI / HSM),
 * pastes the hex signature here, and adds the live TOTP code. The backend
 * verifies both before promoting Paper → Live.
 */
export function ApprovalModal({ strategy }: Props) {
  const [open, setOpen] = useState(false);
  const [signedToken, setSignedToken] = useState("");
  const [totpCode, setTotpCode] = useState("");
  const approve = useApprove();

  // Fresh timestamp captured each time the modal opens — the CEO must sign +
  // submit within the backend's ±60 s replay window (audit V1).
  const timestampMs = useMemo(() => Date.now(), [open]);
  const message = approvalMessage(strategy.experiment_id, timestampMs);

  const handleSubmit = () => {
    approve.mutate(
      {
        experiment_id: strategy.experiment_id,
        timestamp_ms: timestampMs,
        signed_token: signedToken.trim(),
        totp_code: totpCode.trim(),
      },
      {
        onSuccess: () => {
          setOpen(false);
          setSignedToken("");
          setTotpCode("");
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm" data-testid={`approve-${strategy.experiment_id}`}>
          <ShieldCheck className="h-4 w-4" /> Approve
        </Button>
      </DialogTrigger>
      <DialogContent data-testid="approval-modal">
        <DialogHeader>
          <DialogTitle>Authorise Paper → Live</DialogTitle>
          <DialogDescription>
            {strategy.asset} · {strategy.algorithmic_family} · PBO {strategy.pbo?.toFixed(3)} · DSR{" "}
            {strategy.dsr?.toFixed(2)}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div>
            <label className="text-xs font-medium text-muted-foreground">Message to sign</label>
            <code className="mt-1 block break-all rounded-md bg-secondary px-3 py-2 text-xs" data-testid="sign-message">
              {message}
            </code>
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground" htmlFor="signature">
              Ed25519 signature (hex)
            </label>
            <Input
              id="signature"
              data-testid="signature-input"
              autoComplete="off"
              placeholder="128 hex chars"
              value={signedToken}
              onChange={(e) => setSignedToken(e.target.value)}
            />
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground" htmlFor="totp">
              TOTP code (6 digits)
            </label>
            <Input
              id="totp"
              data-testid="totp-input"
              inputMode="numeric"
              maxLength={6}
              placeholder="000000"
              value={totpCode}
              onChange={(e) => setTotpCode(e.target.value)}
            />
          </div>

          {approve.isError && (
            <p className="text-sm text-destructive" data-testid="approve-error" role="alert">
              {approve.error.message}
            </p>
          )}
        </div>

        <DialogFooter>
          <Button
            onClick={handleSubmit}
            disabled={approve.isPending || signedToken.trim() === "" || totpCode.trim() === ""}
            data-testid="approve-submit"
          >
            {approve.isPending ? "Verifying…" : "Sign & Deploy"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
