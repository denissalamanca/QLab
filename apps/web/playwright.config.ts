import { defineConfig, devices } from "@playwright/test";

// E2E exercises the full CEO approval flow against a running frontend (5173)
// which proxies to the control-plane API (8000). Start both before `npm run e2e`:
//   uvicorn apps.api.main:app --port 8000   (with a seeded registry + test key)
//   npm run dev
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: "list",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:5173",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: process.env.E2E_NO_SERVER
    ? undefined
    : {
        command: "npm run dev",
        url: "http://localhost:5173",
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
      },
});
