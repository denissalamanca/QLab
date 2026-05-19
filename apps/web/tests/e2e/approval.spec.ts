import { expect, test } from "@playwright/test";

import type { Strategy } from "../../src/lib/api";

// A representative CPCV-validated strategy awaiting sign-off.
const STRATEGY: Strategy = {
  experiment_id: "11111111-1111-1111-1111-111111111111",
  asset: "EURUSD",
  algorithmic_family: "cusum",
  agent_version: "agent_6@e2e",
  timestamp: "2026-05-20T00:00:00Z",
  num_events_triggered: 750,
  orthogonality_score: 0.12,
  brain_1_recall: 0.82,
  brain_2_log_loss: 0.41,
  pbo: 0.03,
  dsr: 1.42,
  is_deployed: false,
  status: "completed",
};

// 128 hex chars — a structurally valid (test) Ed25519 signature.
const FAKE_SIGNATURE = "ab".repeat(64);

/**
 * Exercises the CEO approval flow end-to-end through the UI. The control-plane
 * API is intercepted so the test is deterministic and needs no live backend or
 * real private key; the request bodies are still asserted to match the §11.1
 * contract.
 */
test.describe("CEO approval flow", () => {
  test("approves a strategy and removes it from the awaiting list", async ({ page }) => {
    let approved = false;

    await page.route("**/api/v1/registry/strategies", async (route) => {
      await route.fulfill({ json: approved ? [] : [STRATEGY] });
    });
    await page.route("**/api/v1/execution/approve", async (route) => {
      const body = route.request().postDataJSON();
      expect(body.experiment_id).toBe(STRATEGY.experiment_id);
      expect(body.signed_token).toBe(FAKE_SIGNATURE);
      expect(body.totp_code).toBe("123456");
      approved = true;
      await route.fulfill({
        json: { experiment_id: STRATEGY.experiment_id, deployed: true, message: "ok" },
      });
    });

    await page.goto("/");
    await expect(page.getByTestId(`strategy-row-${STRATEGY.experiment_id}`)).toBeVisible();

    await page.getByTestId(`approve-${STRATEGY.experiment_id}`).click();
    await expect(page.getByTestId("approval-modal")).toBeVisible();
    await expect(page.getByTestId("sign-message")).toHaveText(`afml:approve:${STRATEGY.experiment_id}`);

    await page.getByTestId("signature-input").fill(FAKE_SIGNATURE);
    await page.getByTestId("totp-input").fill("123456");
    await page.getByTestId("approve-submit").click();

    // After success the modal closes and the list refetches (now empty).
    await expect(page.getByTestId("approval-modal")).toBeHidden();
    await expect(page.getByTestId("strategies-empty")).toBeVisible();
  });

  test("surfaces a 403 when the signature / TOTP is rejected", async ({ page }) => {
    await page.route("**/api/v1/registry/strategies", async (route) => {
      await route.fulfill({ json: [STRATEGY] });
    });
    await page.route("**/api/v1/execution/approve", async (route) => {
      await route.fulfill({ status: 403, json: { detail: "invalid or missing TOTP code" } });
    });

    await page.goto("/");
    await page.getByTestId(`approve-${STRATEGY.experiment_id}`).click();
    await page.getByTestId("signature-input").fill(FAKE_SIGNATURE);
    await page.getByTestId("totp-input").fill("000000");
    await page.getByTestId("approve-submit").click();

    await expect(page.getByTestId("approve-error")).toContainText("invalid or missing TOTP code");
    // Modal stays open so the CEO can retry.
    await expect(page.getByTestId("approval-modal")).toBeVisible();
  });
});
