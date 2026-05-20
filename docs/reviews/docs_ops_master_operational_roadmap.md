# **Master Operational Roadmap: Autonomous Quant Lab (M0 \- M7)**

**Prepared for:** Claude Code (Operations & Deployment Execution)

**Objective:** Operationalize Phases 0-9 into an autonomous, agentic ML trading firm.

**Execution Rule:** Strictly sequential. One milestone per PR. Do not advance until DoD is met.

## **M0: Genesis & Operator Bootstrap**

**Goal:** Establish the secure environment, CLI execution layer, and infrastructural sanity checks.

* **Deliverables:**  
  * Build afml CLI entry point using typer.  
  * afml enroll-ceo: Generate Ed25519 keypair and TOTP seed. Persist securely to macOS Keychain/config. Output QR URI once.  
  * afml doctor: Scan for local Dukascopy .parquet files, verify Redis connection, verify SQLite registry reachability, and check Keychain secrets.  
* **Audit Injection:** afml doctor must not just check for files; it must verify the data contains sufficient "warm-up" history (e.g., ![][image1] FFD length \+ 100 bar feature shift) to cold-start the agents.  
* **DoD:** CLI works. Keychain stores secrets. Control Plane boots and accepts a TOTP token via the /approve endpoint. doctor exits 0 on healthy system, 1 with specific errors on missing data/services.

## **M1: The Historical Ingestion Sweep (Data: 2020 \- 2025\)**

**Goal:** Run the core mathematical engine (Phases 1-6) on the historical dataset to populate the Alpha Registry with mathematically certified strategies.

* **Deliverables:**  
  * Build the ResearchHarness script to iterate through the 14 assets using **only 2020-2025 parquet data**.  
  * Sweep hyperparameters across CUSUM, Bollinger, and Donchian primary alphas.  
  * Save all trials (successes and MDA-failures) to the Alpha Registry to build the DSR penalty (![][image2] count).  
* **Audit Injection (The Stable Plateau):** The harness must *not* select the highest-performing hyperparameter combination. It must programmatically evaluate the grid and select a point in a **stable variance plateau** to prevent curve-fitting.  
* **DoD:** Sweep completes on all 14 assets for the 2020-2025 interval. Registry contains ![][image3] trials. CPCV, PBO, and DSR are calculated and logged for the surviving models.

## **M2: Model Persistence & Artifact Lifecycle**

**Goal:** Serialize the M1 mathematical outputs so the live execution agents can load them deterministically.

* **Deliverables:**  
  * Build ModelStore to save/load Brain-2 weights, calibrators, optimal FFD ![][image4] parameters, selected ONC cluster specs, and Phase 7 GMM arrays.  
  * Bundle these artifacts under the specific strategy\_id (SHA-256 hash) matching the Alpha Registry.  
* **DoD:** A model trained in M1 can be saved, flushed from memory, reloaded, and output identical Brier scores and probabilities on a test array up to ![][image5] precision. The afml doctor command verifies that deployed registry rows have intact, loadable model bundles.

## **M3: The 2026 Out-of-Sample (OOS) Walk-Forward**

**Goal:** Prove that the strategies discovered in the 2020-2025 data survive strict out-of-sample forward exposure before trusting them in live agents.

* **Deliverables:**  
  * Build an OOS\_Validator script.  
  * Load the approved models from M2.  
  * Feed the unseen **2026 parquet data** through the pipeline (Tick ![][image6] Info Bars ![][image6] Features ![][image6] Brain 2 Probability ![][image6] Execution Sizing).  
  * Log the realized 2026 OOS Sharpe Ratio against the projected CPCV Sharpe Ratio from M1.  
* **DoD:** Pipeline runs 2026 data without retraining. Any model whose 2026 OOS Sharpe falls beneath the M1 Deflated Sharpe Ratio (DSR) threshold is automatically quarantined in the registry.

## **M4: The Autonomous Agent Runtime**

**Goal:** Transform the static mathematical libraries into 8 distinct, always-on asynchronous background workers communicating via Redis.

* **Deliverables:**  
  * Build core/orchestrator.py to manage agent lifecycles, error isolation, and heartbeats.  
  * Wire Phase 1-8 logic into Agent1 through Agent8 daemon scripts.  
* **Audit Injection:** \* Agent 8 (GSADF) must *strictly* bind to NEW\_INFORMATION\_BAR events and run inside asyncio.get\_running\_loop().run\_in\_executor() to prevent event loop blocking.  
  * Agent 7 must use asyncio.Lock() when receiving BET\_SIZED events to prevent margin-scaling race conditions.  
* **DoD:** Synthetic ticks published to Redis flow through all 8 agents seamlessly. Heartbeats are logged. CPU does not bottleneck during Agent 8 GSADF computations.

## **M5: The Control Plane Awakening**

**Goal:** Wire the React UI and FastAPI control plane to the live Redis bus to observe the M4 agents.

* **Deliverables:**  
  * Route Alpha Registry data to the React dashboard (showing the surviving 2020-2025 models and their 2026 performance).  
  * Wire the Cryptographic /approve endpoint to emit a CEO\_APPROVAL event to Redis (which Agent 7 listens for to activate a strategy).  
  * Wire the /emergency/flatten endpoint to emit an EMERGENCY\_FLATTEN event.  
* **DoD:** The React UI successfully renders CPCV charts. Approving a strategy in the UI requires the CEO TOTP and successfully updates the Registry and alerts Agent 7 over the live Redis bus.

## **M6: The Broker Bridge (Paper Trading Integration)**

**Goal:** Connect Agent 7 to the MT5 Paper Trading simulator.

* **Deliverables:**  
  * Complete mt5.py BrokerAdapter using the Wine/VM socket bridge.  
  * Route live MT5 ticks into Agent 1's ingestion queue.  
  * Agent 7 executes dynamic lot sizes based on Phase 7 continuous bet sizing math.  
* **Audit Injection (State Rehydration):** Upon startup, Agent 7 must query MT5 for active open positions to rehydrate its concurrent trade count (![][image7]) before sizing new bets.  
* **DoD:** Live MT5 ticks generate Information Bars. Agent 7 successfully dispatches a dynamically sized paper trade to MT5. Restarting Agent 7 successfully fetches the active trade and scales the margin of the next trade appropriately.

## **M7: The Agentic Soak & Disaster Drill**

**Goal:** Final multi-week live paper-trading soak and autonomous safeguard testing.

* **Deliverables:**  
  * Dockerize the entire stack (api, web, redis, agents).  
  * Run the system live on paper money.  
  * **The Disaster Drill:** Inject a synthetic "Flash Crash" tick array into the live feed.  
* **DoD (The Ultimate Test):** 1\. The synthetic flash crash must trigger Agent 8's GSADF break detection.  
  2\. Agent 8 must emit MARKET\_REGIME\_BREAK.  
  3\. Agent 7 must autonomously intercept this event, halt new orders, and execute a flatten command to MT5 to protect capital without CEO intervention.  
  4\. System runs flawlessly in Docker Compose.

**CLAUDE:** Acknowledge this roadmap. Once acknowledged, begin executing M0.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAAzklEQVR4XmNgGKFASkpKRF5e/j8UX0OXl5OT60cXQwFABdpQzVEwMSB7LhA3A3ERUH6FgoJCIbIeOAAqyAFplpWVlUITXwrEn/DaDlTwGqQZTWwfUFMskM4C2toIxB3I8nAAdTK6Zk8QDTSgDso3RJaHA6jmq+jiIAAKUHQxOAA6xwCqOQhdjiAAhSJQ4z9paWlhdDmCAKjxPd7QxAdATgbaLgBlYw8UXAAWysbGxqzAePZDl8cLQJq1tLTYgPQJdDmiANDZ9uhio2AUgAAApkAw0nj61PYAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABKUlEQVR4XmNgGAUUAVlZWVN5efn/WPBumBoFBYUCZDkgPxLZDBSgoqLCLicnNwOkEEjPBwqxoClhhBr0TVFR0Q1NDgOwABV+AeJf4uLi3MgSQLEsIP6ELIYXAJ1tDrX5CLK4jIwMJ1DsAdC1xsjieAFQwxaoYY4wMaAFHkB+M7I6ogBQ00eQYUADOKD8WiD+i66OKAB11X9g4OoD6TswPjBi+NDVEgLMMM1Al1mABIDsZKjYVXTFeAHQNWZQjRdhYtCAfwsSR1ZLEABjaifUVeZo4kogcWlpaRlkcbxAHpK+PjNgJlRwWAINbUMXxwmgXqxGFwcBoPgzqIFp6HIoAKjoG9RVoGTxHYh/S0lJccHkgd6+JQ8Jt1dA/B6kHuhlYWQzRsEoGJIAABRCW0oCXUcRAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAaCAYAAADsS+FMAAADNUlEQVR4Xu2WS0hUURjHrexJBFFGjM7cOzq1sKLAVc9FSYREqx4UEUG4kTa56g21qgiKioqCHraIzKJFi8pFtAoqatVCKNKKQMPooUWK2e+b+S4evpzxziiCcf/w597z/x7nO+eee84pKooQIUKEAlFeXj7f87x+S0zjAx/aS429wUkxakgkEmvo+xK8C2tisdg06yPwfX8P9iae161tSKRSqckEH5WBkuBRZWXlJOujtkYK2mhtFvg9g6etPhxUVVVNpIZf8DI8pB+lxfolk8lV6O9gHTwDH1PLXOuXEwzyvXRQVlaWMno1eocU4+phQOw1YnsoZp215Yli8nTDtpKSkuki8N4k9ZL7WOBEu1m0gbC01mK1nCChLwEyIcY0Dv0r+majhwa/YUIGQhdbaU6w9jCoqKiYI/VpjdWi8byg7XuBH+0/duCyQq2WEyS8qol3BBrtN/Cm6zccyBclXz1sh/utPV+Qo0drXu5oMmFdrh/2WtHj8fgGV88KnFslgC8QlzazuVNm3vqNBOhnl5dZuiesLR/owO0vIVq70bapXufqWREk1t/lBeyVNpvRYus7kqCPTunL6tnA112I/w+tba1rY6+bquP44Oqy4YvO85SrZ0UwGbBG2gSu0Ha39R0uyD2TvAdhW7ajMQyIf6mDXKCS7G9S80fXD/sm1U+6+qBw7hmtrq5a+I0nBPiyMXJ+h68LOZ1cBHsBfEWzWDRtf3L9WOxbROe519UHBU6N4mw3GC+zhGXm17t6ISDPbXifPlZaWxjIcU/8ed+5L3A3mqGDl4HuE03bPQORaW23+gx9vOP4GfbJ8nV1go9r8mZXzwfErpZ4eMPa8gG1PZRaeF4MNF1lUp/o6aM/aA9EpsdxRDTZU1x9UIijn+W2qAORDq5YWy5Q3BNiOtkTZltbIaC+W1rHskDj/YBOxINAk3c7GVKH1f4BDj9hF/zmZa65vXTqO3a58UmiDvgF9jnhow6+7CIvc6l6C5/q+x3rh5aEv2EDfA7PMqwp1m/MQ045eJgBnmNy5ll7AAa/BJ96fLdb26hDjkz5RcKwtLR0lo3/r6DLuD8sbXyECBEiRIgwtvAXqn4avRQiwn4AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABL0lEQVR4XmNgGAVUA3JycuXy8vL/YRgmrqCgIACUs0FWSwxgFBcX54YZpqWlxQakJYHiTEADNUByioqK4uia8AKoYatBbKAhAUD2aSB9C+i6NHS1eAFQgyDUsBwovwzIfgjFPejqMYC6ujov0PnyIDZQcz9Q0xsgkxGEZWRkOIE0M1BcG0gzgdQi60UBQOfbAzX/A+LfQA0uQPo8EK9DVgN1rRWyGDpgBCqYB8QXYAJA9j2QF4Gu1ENWSBBAwwAe/VCxM+hiRAGQJnSNQP5fdDGiANSwT1jE3iGLEQWgGrejiwEDewOQyQKk65Dl8AKQRmBMbgKxRUVFeYD8fVAxcyC9RJ6YNAUDUE2Pgfg+kN0BFGIBsn+AvE5ySocBoEHuMDYogQL5Fsjyo2AUDAYAAFyfT5oweqpEAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAaCAYAAAAT6cSuAAAB5UlEQVR4Xu2WO0sDQRRGJQZBtBENgbwDsUltQFArQSxsfWJhIxY2WosgaGGl2NjqD7C0sdZCrYI2NpaCrxBERUVQz8isTK6bdUHzAOfAZXbv/Wbmu7ObR0ODxWKpKYlEoieZTO7LfD0Ri8Wi+NzDZz4ej/fJuhtBJgwz4V2FLNYLqVRqFH9FxiX8TmuvQan7gu5ziMe5DNZzc/iMKG94XXVymUwmpHLpdHrA1LpSz83ha01544kNibzKbZs5V/w2x3vfTHTKfCXB14XyxhPsEnlfnn0JqS8QL8QrsRMOh1ukphKw16VuLifyP3r+xEvIo5+k9sbYa+TWyT2aukrBZ21DeROvZcDLcwleQl2717cBNuvmvqi+mkuEBqFQqNVvZLPZJjlfoj18HSbXeS/PJZQTqvfcaY6TO2J8Jgrq20pqTZz1/ASHdSznS9BNEE942GJcZs68M19qv1FOSG5F5Vm0X9ZqjfZ8KvPf8GhuUeX5PUnKWjVRv8cccJtzr66VL/Izps6Vcs1Fo9F2t0XIHfo6tT+AvVPa37mT4/qM9JipK4HuRxA9ETfElY5b4k7o1N+zAoudaP2BWa8G7PlAXBO7xDNepqTmNzSy4CBjQBaqQSQS6WD/ORrbJGZl3WKxWCyW/8IHz2ur010gFqIAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAZCAYAAADe1WXtAAAAh0lEQVR4XmNgGAWjYOCBnJycC7oYxUBeXv4jEBuhi1MElJSU1ICGfkYXpxgADS1WUFCQQBeHA6CkBlCRJhl4M8jFMjIyQuhmgmz9TwkGOuoUuplkA6CBZ4CulEYXJwsADVIFGvgVXZwiADRwEjCtzkAXpwgADb2GLkYxUFRU1EcXGwWjgIYAAKRyLa8bu1G/AAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABLklEQVR4Xu2Tq0oFURSGx+ATiJa5XxCxGEbQIiImg88gnC6CwW6zCKJZi0ZBMBkVfAOrT2ASLCrK8VvD3rL5VZxy2vlgMWf+f91mz5koGgVVVWV5nh/+FkVRrGr+aInjeCpN05W6rme81rbtZJZl82HeD1h3g3WvuQ5dfFJ047xL4kRrOig6cMl3SZI0oWeT8d6sIZsloddB0baZXPfU8+DfE2eqd9hkW5ufE+p5yDknZ0t1M05tuuoKOceqRZxwivFOfKin8HibqlnXB5tO7KrXCwpfXYM19Xrhiv99foMDXFatdwOK55qmmVa9ezXWoCzLBfU8FC+R86J6Bye77ra4UM+D90gMVP/G/U2HTDric571Ovf76M9h7p+wyY4/D+KWeCKu+BoXNXfMmJAv6RROn+12BRMAAAAASUVORK5CYII=>