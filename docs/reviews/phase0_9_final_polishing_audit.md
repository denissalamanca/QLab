# **Institutional Audit Report: Quant Lab Final Polish (Phases 0-9)**

**Prepared for:** Claude Code (Development Agent)

**Author:** Lead Quant & Product Manager (Lopez de Prado Framework)

**Status:** HOLD (Final Runtime Patches Required)

## **Executive Summary**

The AFML engine and the Human-in-the-Loop Control Plane are structurally complete and theoretically sound. The Replay Attack mitigations, execution thread-locks, and type-casting serializations are properly implemented.

However, a live, asynchronous trading environment is brutal. I have identified **four critical runtime-killer vulnerabilities** that will cause the system to crash, lag, or lock the CEO out during live execution. You must patch these final issues immediately.

## **1\. Mathematical & Computational Vulnerabilities**

### **1.1. Vulnerability 1: The Scikit-Learn GMM .cdf() Trap (Phase 7\)**

* **Location:** src/afml/execution/bet\_sizing.py  
* **The Flaw:** When the Shapiro-Wilk test fails, the architecture correctly instructs Agent 7 to fit a GaussianMixture (GMM) to the probabilities. However, sklearn.mixture.GaussianMixture **does not have a .cdf() method.** It only has .score\_samples() (log PDF) and .predict\_proba(). If your bet sizer attempts to call gmm.cdf(z\_i), the pipeline will instantly throw an AttributeError: 'GaussianMixture' object has no attribute 'cdf' and fail to execute the trade.  
* **The Refactoring Directive:**  
  1. You must mathematically compute the CDF of the Gaussian Mixture manually using scipy.stats.norm.cdf.  
  2. The exact formula is the weighted sum of the component CDFs:  
     ![][image1]  
     *(Where ![][image2] are gmm.weights\_, ![][image3] are gmm.means\_, and ![][image4] are gmm.covariances\_).*

### **1.2. Vulnerability 2: The GSADF "Tick-Death" CPU Bottleneck (Phase 8\)**

* **Location:** src/afml/monitoring/pipeline.py & Message Broker Wiring  
* **The Flaw:** The Generalized Supremum ADF (GSADF) test is computationally brutal, requiring hundreds of nested expanding-window OLS regressions. If Agent 8's listener executes the GSADF function synchronously on *every single incoming tick* (NEW\_TICK event), the Python event loop will completely bottleneck. CPU utilization will hit 100%, memory queues will overflow, and Agent 7 will experience massive execution latency.  
* **The Refactoring Directive:**  
  1. Agent 8 must **never** run GSADF on raw ticks.  
  2. Agent 8's listener must be re-wired to trigger only on the NEW\_INFORMATION\_BAR event dispatched by Phase 1\.  
  3. Wrap the GSADF computation inside an asynchronous thread/process offload (e.g., asyncio.get\_running\_loop().run\_in\_executor()) so the main message broker loop is never blocked by the monitoring agent.

## **2\. Security & Control Plane Vulnerabilities**

### **2.3. Vulnerability 3: Ephemeral TOTP Seed Lockout (Phase 9\)**

* **Location:** src/afml/crypto/totp.py & src/afml/control\_plane/security.py  
* **The Flaw:** If the TOTP (2FA) base secret seed is generated randomly in-memory every time the FastAPI control plane boots up, the QR code/secret the CEO scanned into their Authenticator app will immediately invalidate on the first server restart. This will permanently lock the CEO out of the "Emergency Flatten" and "Deploy" endpoints after a reboot.  
* **The Refactoring Directive:**  
  1. The TOTP seed must be deterministic and persistent.  
  2. Check the local macOS Keychain (or a .env / encrypted local SQLite config table) for an existing AFML\_TOTP\_SEED. If it does not exist, generate it, display it to the console *once* for the CEO to scan, and securely persist it.  
  3. The FastAPI app must load this exact persisted seed into memory on every boot.

### **2.4. Vulnerability 4: Vite/FastAPI Strict CORS Preflight Block (Phase 9\)**

* **Location:** apps/api/main.py  
* **The Flaw:** The React dashboard runs on Vite's default dev server (http://localhost:5173 or 127.0.0.1:5173), while FastAPI runs on 8000\. By default, modern browsers will block all GET and POST requests to the API with a CORS Preflight violation, meaning the Dashboard will render but remain entirely empty.  
* **The Refactoring Directive:**  
  1. You must explicitly configure fastapi.middleware.cors.CORSMiddleware in main.py.  
  2. Explicitly define allow\_origins=\["http://localhost:5173", "http://127.0.0.1:5173"\] (do not use \["\*"\] in an institutional app).  
  3. Set allow\_credentials=True, allow\_methods=\["\*"\], and allow\_headers=\["\*"\].

## **3\. Execution Orders & Final Sign-Off**

1. Construct the manual CDF summation for the GMM in bet\_sizing.py.  
2. Offload GSADF to run\_in\_executor and bind it strictly to Information Bars.  
3. Persist the TOTP secret to the keychain/config so the authenticator app survives reboots.  
4. Apply the strict CORSMiddleware configuration to the FastAPI entry point.  
5. Provide the unit test specifically validating the custom GMM CDF calculation.  
6. **Report back when complete.**

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAaCAYAAABctMd+AAABWElEQVR4Xu2Tu0rEUBCG1xUEb4VCDOQeUgSi4gtYbCF2gjYigq8gYmnjSwhaiA9g5RuojyCIFxAWX0ArbZRdv5FkOY4hK2Q794MhZ/6Z+c/Z5GyjMWTI/2NEC5o0Tae1VkkQBPNhGF4R7ZLadZIkc7KmvkE86Z5KGDjLn5+YbRZ6FEV7aO9Zlo3ldTHvFvW+eJ63WKzVYJP8hc0ODU33fGNZ1hQH2dJ6D8dxJszBOI6XJGeoZbSJ+YOZCxzguGzTHvJqiNsix/S8ZGAUo2Wl9Qeje+LUyNvaPDdumtqfwOiS4Qsj75jmvu8vkL8WuUD/CbGC/kYcmbVfyDDxTHzk+YGsiTtMdnU/l2E87+sS27r+A9u2J/mQqxjNFJrcJvI1s08j5q7relqvDZvPhvn/ZOBwq9b5ZTtaHwic+kZrtZDvgOk+p454dnS9FnIFMX2Ud80VdXR9SCVfHUdLS7R2BD0AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAaCAYAAAC3g3x9AAABNElEQVR4XmNgGAWjYLgDBQWFmXJycmXIYrKysspAsXpkMaIAUJOLvLz8fyBtjCwOFHsCxM3IYkQBoEEnQQaii4PExMXFudHFQcDY2JgV6CsJdHEwAGr8g8tAdDEYAMqVAHExujgYgDQC8TtkMUVFRX18BuIFUAO3I4sBvVMIFPsLlQ9ClsMLgIqDQQaCwhFJLAkqNgOIrYHsj0hyPUCKCUjfgYmhAKDEVSC+C3TRLZAiIH4LxIpSUlIiUEN3yMjIcMLUq6io8AGTkylIDtkcOABJAA3LALGBml2BfEmYHFBcA6ESAeQhEYLbQHQxQgCo5xPQskPo4mBApoGgoMDMQUBbLICS39DFCQGgni+ghI0uDgKMoPyKLogNACNGFxR20Mhaji5PMgAlIaBBd4D0CnS5EQwAG0RKlqIJDMUAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAZCAYAAADaILXQAAABbklEQVR4Xu2UOy8EURiG1yWRaHQkY8xldwv+BCHUIlEIElGI6BSIyk8QCtH4ASqlaChQSbbR8gtcii1IRHbXc+RMHK/dNTGbiMSTvJmZ9/vOey4zmVzun18hDMOa1T06RWdRFJ1zvUQ3qJL00N6h45sSBMGhHfysNUM+n+9hsmvTUygUBrT+LQwctRNsay2BRYxTn1U/FTa8SsiE1hLYwaZ6qSB4yDnbuhC+ol5qCD62E5Q560DrmYjjuI/gOzvBkdYzw9ZXOfd59RvheV43/Vvqf4HgSVZcUT+BkCX1GLPY7D29Y86YpotisdiltQTCN9RLQyfBj2o6tJuJ0ZjrseodvAcmnXb8z9BwQOOI+pY2aldm61wHE5PnNZ4XCD7h/tbp/8CsGD2hV1S1qtWTO45fQa8db2p7bq1lmHB2MKN+SwgbHUlWCJ5Cc+q3BI5j1/f9fvV/DF/Hvn2J6+hF65lgtcuEllGJf9Gw1v8+b7UaYtusvNyyAAAAAElFTkSuQmCC>