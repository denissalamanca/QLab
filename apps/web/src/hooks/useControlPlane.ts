import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  api,
  type ApproveRequest,
  type ApproveResponse,
  type FlattenRequest,
  type FlattenResponse,
  type Strategy,
} from "@/lib/api";

const STRATEGIES_KEY = ["strategies"] as const;

/** Poll the strategies awaiting sign-off (refresh every 15s). */
export function useStrategies() {
  return useQuery<Strategy[]>({
    queryKey: STRATEGIES_KEY,
    queryFn: api.listStrategies,
    refetchInterval: 15_000,
  });
}

/** Approve (Paper → Live) — invalidates the list on success. */
export function useApprove() {
  const qc = useQueryClient();
  return useMutation<ApproveResponse, Error, ApproveRequest>({
    mutationFn: api.approve,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: STRATEGIES_KEY });
    },
  });
}

/** Emergency flatten kill-switch. */
export function useFlatten() {
  const qc = useQueryClient();
  return useMutation<FlattenResponse, Error, FlattenRequest>({
    mutationFn: api.flatten,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: STRATEGIES_KEY });
    },
  });
}
