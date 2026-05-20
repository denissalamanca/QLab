# **Institutional Audit Report: Quant Lab Full-Stack Integration (Phases 0-9)**

**Prepared for:** Claude Code (Development Agent)

**Author:** Lead Quant & Product Manager (Lopez de Prado Framework)

**Status:** HOLD (Final Production & Security Patches Required)

## **Executive Summary**

The completion of Phase 9 (The Control Plane) officially links the mathematical AFML factory to the Human-In-The-Loop execution engine. The React architecture, Vite tooling, and FastAPI routing schemas represent top-tier software engineering.

However, an uncompromising integration audit reveals **five critical vulnerabilities** at the boundaries of the system. These vulnerabilities span async database blocking, cryptographic security flaws, data serialization crashes, and execution race conditions.

You are directed to execute these five final refactoring directives to harden the system for live capital deployment.

## **1\. Control Plane & Security Vulnerabilities (Phase 9\)**

### **1.1. Vulnerability 1: Cryptographic Replay Attacks on Execution**

* **Location:** src/afml/control\_plane/security.py & apps/api/routes.py  
* **The Flaw:** The POST /api/v1/execution/approve endpoint verifies the CEO's Ed25519 signature. However, if the payload only contains the strategy\_id and the signature, the endpoint is highly vulnerable to a **Replay Attack**. A malicious actor (or a network retry glitch) could capture the signed payload and repeatedly resend it, causing the system to deploy or flatten positions continuously.  
* **The Refactoring Directive:**  
  1. The API payload must enforce a strictly increasing nonce or a timestamp.  
  2. The frontend must include the current UTC timestamp (in milliseconds) in the payload, and the Ed25519 signature must sign a concatenated string of strategy\_id \+ timestamp.  
  3. The backend must reject any payload where abs(current\_server\_time \- payload\_timestamp) \> 60\_000 (60 seconds).

### **1.2. Vulnerability 2: FastAPI Event Loop Blocking via SQLite**

* **Location:** src/afml/core/registry/repository.py & apps/api/routes.py  
* **The Flaw:** FastAPI is an asynchronous framework (async def). The Alpha Registry uses sqlite3, which is fundamentally synchronous and blocking. If multiple web clients (or the polling React dashboard) query the /strategies endpoint while the background agents are writing to the database, the synchronous SQLite queries will block the FastAPI ASGI event loop, freezing the entire control plane.  
* **The Refactoring Directive:**  
  1. Do not use standard sqlite3 inside FastAPI async routes directly.  
  2. You must either refactor the read operations in the API routes to use fastapi.concurrency.run\_in\_threadpool, OR migrate the registry repository read methods to use aiosqlite.

### **1.3. Vulnerability 3: Pandas/Numpy API Serialization Crashes**

* **Location:** apps/api/main.py & src/afml/control\_plane/schemas.py  
* **The Flaw:** The Alpha Registry and the ML pipeline heavily utilize numpy.float64, pandas.Timestamp, and numpy.int64. When FastAPI attempts to return these inside Pydantic models (e.g., the stats dictionary containing the DSR and PBO), standard Pydantic/JSON encoders will throw an Internal Server Error: Object of type int64 is not JSON serializable.  
* **The Refactoring Directive:**  
  1. Ensure all Pydantic schemas in Phase 9 explicitly cast numpy types to native Python float and int.  
  2. Implement a custom JSON encoder or utilize Pydantic's @model\_validator(mode='before') to scrub and convert any Pandas datetimes to ISO-8601 strings before they hit the API response boundary.

## **2\. Execution & Monitoring Vulnerabilities (Phases 7 & 8\)**

### **2.1. Vulnerability 4: The Bet-Sizing State Race Condition**

* **Location:** src/afml/execution/pipeline.py & src/afml/execution/bet\_sizing.py  
* **The Flaw:** In the previous audit, we mandated that Agent 7 query get\_open\_positions() to rehydrate its concurrent trade count (![][image1]) upon startup. However, in a live asynchronous tick environment, there is a **Race Condition**. If Agent 7 receives a probability signal from Brain 2 *while* it is actively awaiting an HTTP/Socket response from MT5 regarding open positions, the margin scaler will use an outdated, stale margin count, potentially over-leveraging the account.  
* **The Refactoring Directive:**  
  1. Implement an asyncio.Lock around the bet sizing and order dispatch sequence.  
  2. Agent 7 must strictly sequentialize: Acquire Lock \-\> Fetch Current MT5 Margin/Positions \-\> Calculate AFML Size \-\> Dispatch Order \-\> Release Lock. No signal may be processed simultaneously.

### **2.2. Vulnerability 5: GSADF Stagnant Tick Multicollinearity**

* **Location:** src/afml/monitoring/gsadf.py  
* **The Flaw:** If the live market flatlines completely (e.g., a highly illiquid session where the price does not change for ![][image2] consecutive bars), the lagged differences ![][image3] used in the ADF regression will all be exactly zero. The regression design matrix becomes perfectly multicollinear (singular), crashing the GSADF calculation with an np.linalg.LinAlgError.  
* **The Refactoring Directive:**  
  1. Before performing the GSADF loop, verify the variance of the price array within the current window: if np.var(y) \< 1e-10: return 0.0.  
  2. If the market is perfectly stagnant, the test statistic is 0.0 (no explosive root).

## **3\. Execution Orders & Final Clearance**

1. Implement the Ed25519 timestamp payload signing to prevent Replay Attacks in Phase 9\.  
2. Wrap Phase 9 SQLite reads in run\_in\_threadpool or migrate to aiosqlite to prevent ASGI event loop blocking.  
3. Apply explicit Pydantic type-casting to ensure numpy/pandas artifacts serialize correctly over the API.  
4. Add the asyncio.Lock to the Agent 7 execution pipeline to eliminate margin calculation race conditions.  
5. Add the variance zero-check to the Phase 8 GSADF regressions.  
6. Verify these edge cases with explicit unit tests in your tests/unit/phase9/ and tests/integration/ suites.  
7. **Report back when complete.** The system will then be fully cleared for live deployment.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABLklEQVR4Xu2Tq0oFURSGx+ATiJa5XxCxGEbQIiImg88gnC6CwW6zCKJZi0ZBMBkVfAOrT2ASLCrK8VvD3rL5VZxy2vlgMWf+f91mz5koGgVVVWV5nh/+FkVRrGr+aInjeCpN05W6rme81rbtZJZl82HeD1h3g3WvuQ5dfFJ047xL4kRrOig6cMl3SZI0oWeT8d6sIZsloddB0baZXPfU8+DfE2eqd9hkW5ufE+p5yDknZ0t1M05tuuoKOceqRZxwivFOfKin8HibqlnXB5tO7KrXCwpfXYM19Xrhiv99foMDXFatdwOK55qmmVa9ezXWoCzLBfU8FC+R86J6Bye77ra4UM+D90gMVP/G/U2HTDric571Ovf76M9h7p+wyY4/D+KWeCKu+BoXNXfMmJAv6RROn+12BRMAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABKElEQVR4Xu2SvUrEUBCFF3/3AUQkhvyQVGkDsuJTaCUWvs2C9RZbLWgvWIuC+ADa2FhZWikW6loI6n6jN8s4e29lJeSDIZlzzh0ml3Q6LX8iz/ONNE2/dNkM2oPJbNvMlCzLugT6oWGwgH4pXpIkW9a0SPiV4JEcYPiBDaBf4B9afQYO9wh/Eq7ddo+yrc6gjfH3teaF0Cnhgbzz3HMDr3WG/k73QQi+UDvyXtf1ohv26+7oR7oPwifd6J6D7zIMfc35vTiO13UmhFz+91YN9Jtuu3vpGXam/SByMIqiFY9+JQOrqlriOba+F4LnVhPKslx2251Qt9b3QvDNag1umNTQej7mCD5ZsQHvWIYVRbFqPc283EP680s8Ux9c8q4NCXh9q7W0/GsmMMNTINO+gpAAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACEAAAAaCAYAAAA5WTUBAAAB7klEQVR4Xu1Vu0pDQRCNRsEHoggauPfmeQMBq0jAwkLxAwSDlYU2NjYi2Igi2AYrP0AQS8HGwicRrawsIhaCTX5ArCIqovGM7MhkkvjiJiDkwLA758zOzM7mEp+vgf8C13X7Lcvq0HzdEA6Hx2BFWCEej/dpvS5A8TPTRDESiWS0XhegeI6eghvRes0RjUaHUHiS9txELBbr1nE1A0aflDeHv2wauZNxGsFg0NLcn4Fiu7Ab9hOJRBf8+6+eBI1OQH+FXSl+CtyF5H4EHLzF4ldcDzURCoUWJG/ghzaLJxyhGEzEZcFMsGrzFYEiW1haNE9AsgIlRENzkqfbGj2vC5omTiX3LXDgRXMMaOsmaV5rBH3rVCrVSj4utiZi0vCX2C+D4zg2AuY1LyEKlTyX0DaFvwp7wqTaVMwi+2VA8IY8UAlI8EyJ+AkY+D0EzK1nRGwWdi7CPprA2aTkPgEhY7rMYn+CZMfYH8IOlF2aOLI0n6cpEodz48yZgqO0Bz8AfxDrEe05pgQi8a9M5diHPcIeYLkK+krVKXgJTKSdCmHt1U3QhLE0S85TUEGMOUZ727Ydmgj8aaEPc1OBQKDTV4tmqAB9krTHjfdMwSahp7kJ6NvMewok3kGRN9g13VrrAmWftqeg/xjNNdDAX/EOIMWggX1G8oIAAAAASUVORK5CYII=>