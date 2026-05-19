# **Institutional Audit Report: Quant Lab Full-Stack (Phases 0-8)**

**Prepared for:** Claude Code (Development Agent)

**Author:** Lead Quant & Product Manager (Lopez de Prado Framework)

**Status:** HOLD (Final Production Patches Required)

## **Executive Summary**

The implementation of Agent 7 (Probabilistic Execution) and Agent 8 (Monitoring) completes the AFML factory. The logic governing the Shapiro-Wilk GMM override, concurrent margin calculation, and structural break detection is highly commendable.

However, a live distributed system handling real-time asynchronous ticks is unforgiving. A full-stack integration audit has revealed two mathematical edge cases that will crash the execution engine, and two distributed messaging flaws that will sever agent communication.

You are directed to patch the following four vulnerabilities before we initiate end-to-end live testing.

## **1\. Phase 7 & 8 Mathematical Vulnerabilities**

### **1.1. Vulnerability 1: The Bet-Sizing Division-by-Zero Trap (Phase 7\)**

* **Location:** src/afml/execution/bet\_sizing.py  
* **The Flaw:** The formula to convert Brain 2 probabilities into a standardized Z-score is ![][image1]. Brain 2 is a Random Forest/XGBoost ensemble. If the ensemble is absolutely certain, it can output a probability of exactly 1.0 or 0.0. When ![][image2], the denominator evaluates to 0, throwing a ZeroDivisionError and crashing the execution agent right when it finds the best possible trade.  
* **The AFML Refactoring Directive:**  
  1. You must bound the probabilities slightly away from absolute certainty before passing them to the Z-score equation.  
  2. Implement an explicit clip: p\_i \= np.clip(p\_i, 1e-5, 1 \- 1e-5).

### **1.2. Vulnerability 2: GSADF Minimum Window (![][image3]) Collapse (Phase 8\)**

* **Location:** src/afml/monitoring/gsadf.py  
* **The Flaw:** The Generalized Supremum ADF test evaluates explosive roots using nested loops of expanding windows. If the minimum window size (![][image3]) is too small (e.g., ![][image4] bars), the underlying OLS regressions will throw exceptions due to insufficient degrees of freedom, crashing Agent 8\.  
* **The AFML Refactoring Directive:**  
  1. You must explicitly define and enforce a minimum window size ![][image3].  
  2. Ensure ![][image5]. If the total series length fed to the GSADF module is less than ![][image3], it must safely return 0.0 or False (no bubble detected) instead of attempting the regression.

## **2\. Full-Stack Integration Vulnerabilities (Phases 0-8)**

### **2.1. Vulnerability 3: Distributed Serialization of ML Artifacts (Phase 0/7 Bus)**

* **Location:** src/afml/core/broker.py & inter-agent messaging pipelines.  
* **The Flaw:** You have architected a robust Redis pub/sub messaging broker. However, the data flowing between Phase 3 (Feature Matrices), Phase 5 (Probabilities), and Phase 7 (Execution) heavily utilizes numpy.float64, numpy.int64, and pandas.Timestamp datatypes. The standard Python json library cannot serialize these types, which will cause the message broker to instantly crash when Agent 5 tries to send a prediction array to Agent 7\.  
* **The AFML Refactoring Directive:**  
  1. Implement a custom JSON encoder/decoder in your core broker module.  
  2. The encoder must convert pd.Timestamp to ISO-8601 strings, and all numpy numerical types to native Python float or int before publishing to Redis.

### **2.2. Vulnerability 4: Execution Agent State Rehydration (Phase 7\)**

* **Location:** src/afml/execution/pipeline.py & src/afml/execution/risk.py  
* **The Flaw:** Agent 7 calculates bet sizes based on *concurrent active positions* (![][image6]). If the system restarts, updates, or crashes and reboots, Agent 7 starts with an in-memory active position count of 0\. It will then drastically oversize the next incoming trade, breaching FTMO/ESMA margin limits because it "forgot" about the 5 trades that are currently open in the broker.  
* **The AFML Refactoring Directive:**  
  1. Agent 7 cannot rely solely on in-memory state for ![][image6].  
  2. Upon initialization, Agent 7 *must* query the BrokerAdapter (via a new method, e.g., adapter.get\_open\_positions()) to rehydrate its true concurrent active positions before processing any new probability signals from Brain 2\.

## **3\. Execution Orders**

1. Implement np.clip in the Phase 7 bet sizing math.  
2. Enforce the ![][image3] minimum window logic in Phase 8 GSADF.  
3. Patch the Redis message bus to handle NumPy/Pandas serialization.  
4. Add the get\_open\_positions() rehydration logic to Agent 7's startup sequence.  
5. Provide the unit tests verifying the Zero-Division fix and the NumPy JSON serialization.  
6. **Report back when complete.**

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHYAAAAkCAYAAABc4c7tAAAFIklEQVR4Xu2bW4gcRRRA142vqESiris7u1OzD0XWV3DVGF8ofuiHgolBwRiQKEIiihIjSFBX90tJFEVEY0hE/NAfESGSj2gIfogPIphEDRIU/ciHig/ErBEf525X7d699Mz0zvQMOzN14FJV997q6a7b1VXVXdPVFamLYrF4r3NuU6lUOtHaAtjXIYNkF+C/GrnN+kTmCSMjIycQrJ9CmfxH2q7BdhCZJPif9PX1nWHtkXkEgVqF/KfKGwYGBoa1TwDbV1YXyQl6y7s08NdDQ0M8CYvbye+yPnOBY3ysA0t5JfKc9gng96U8gjmHp/v7+xdae6RG6El9ktLA7/EI7fH5VTT2VZKnsQs0+lm6TjWo/7MOLPVvofy29gmg36LyvyJ7tT1SI4VC4XSSBTTob0FHUJ+QiY/PS+97WNkeoTxRTsSH9JAOLPnlyLZQLgfH3qPrRepEAmcC8QsTmZMkPzo6evyMZzao/4Y+np8d36N9AuZ3d8TA5ohv0L9UeapxSW/lMfrFjGc2pJ4J2MTw8PCZkpdxnPIaZTus8vuQf0O5aTDeXEBybCgzPl2izC2LBIFe9Q7pNmS/V3f7ZcvmWc4Z8QHcwY3xEOnrQU9+I/KDKm9AnkVexfd5+c1gawpy8UE4gbWkd5MusX6tCNdyJK1B0d+BfhHpY9bWFtAzzyeIr3CBK2Ri4ZLZW01jAfWOzkV4Spxmj5EnMuvldz60egH9MmSCa15sbW2Hn/4/afUE/2r0N1p9pAVg3DiHO/hbqxfQ/40csPpm4dRQ0cli26Uq0iN5JN1s9XNFJmBzEVs/kjNOTcsDPHpLVlcNe4dVE3n022NEcoDYLaWB/5TlgLwaIx2lvBX5R+yk+2j8s10ys1xk67cCcgNxbePIta0u9trKIq+5xsbGjpM8DTApjYDutZ6enlNEJ5+a/Jrtpdk1WwcaZNzq2p7e3t6TQ75QKPQT1Ju0XSCoGwcHBy+z+lagI4OaBZlUSS+WPOkH1j6fkaCGc68VbvT7ra7R8IQ8lfM+aPW54pKtHTJG3dBqdz/nvbuec6b++i71itXT3YwJH+d9V/jk2Ejk4rqtcr5TT1AFp76xcqwllF+gB+9s1jt05yewEUWGoB5jFRrqP5q2CiCw23MKbNWOQmBf7Mrg11Gkja3+o/ou0tWk3yH3WZ+ABNDqhHoCW1JbdEj3yrlYHw3+D/J7Y1bfsaT1VhroZZnZ64CT/0b7aPD/3uoEH9hLrb4a1bbopMF1nIv9AatvWyRwacELpNlooCtd8m10KrCydid/VNnPC3nZVVFufJPA2qVfMcN2mmpbdPSyM+DP4ymrb0skMEGsTagUdGlU5H2fX6+PIUuMGc/EV5cDEliOv9Tqs+Bmb9GRIE9v0QmphhvoIn5vpdW3HaXktdp4pcCKvkJgp23kPxcxLtNgO2R1gg/s5VafBae26JSSnY3T11BM+Sbski04F1t92yIB9sHdnWIbt7qAS96Jvyn11Ji3jDq3I29pX8prw1go0HsuxPdHl/T63536J0BW5JyLfosO6Z1KL1tn/tC+Xr/V6tqetOBKWftosF1BYz5u9QJBuya8L9dQ51OrqwdXZouOUEx5w4X/pNV1BD6wU8GUnlqht8p4tpnGu94aBGxbkOtS9Lk1rLytcmW26IjN/q8H3UL8n9G6jiEENvRea88DjrvJ6hqB7sncgEP87mfa3lFID/XBreudcCXSliGNRpY5pQp/zewIJKiN6q2RSCQSidTI//Mx8MqfEXmYAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAaCAYAAAAKYioIAAACgklEQVR4Xu2Wu4sTURjFs7LKqiAqSDQkmWSTFQ1aSEpFtPTVCAoWFhaiNv4DIoiFpVZrYa8Igo2wgqvgo7AwlYWNCgoWPhofsIqg8fft3gk3J+MkA5I1y/3BYWbO9907Myc3M5PLBQKBQKCHKIqeVKvVvPppVCqVC+geY6+Wy+WdWh9lxrmhI9zYLGqjqjb8DcadRY/QccbdQb+xx7VvJOHXPpZbCOdKlmAKhcIqF0QHVhtW9NP3Rp6swRDoCXrfqm9zqNdDo9FY4R8Xi8UpJtzoe/8LWYLhHiZstaAZrdkc9Xp9g/odONFJmt6hFodj9n9k/4dNODk5uVn7F5sswdDbcL23tOb8Hep3oHjZba3xGSmv9Wr9l5tgzwLGfc4inSONLMHQs8/1Xk+otZnrqPrzlEqlbWzGbN8apRx783WD/oJ/vBjEwXAtNa0p9O5ywdzQmvMPq99FPp9fnRKMf9yq1Wol3xs2cTA8B+taU1i9W9zKuKk183k77Va/CwZeovGb7zHpGQ1mEJrN5nK7oCzSOdLwgpnSWgLL6J1DD7Rgc1S8x0YiND1Fd8WbQa99bxDcR1Q7i3SONOJgBn0x0HubAJ4n+OnnpWG/u8Dp2GOix/5A9q/Zsus72RCIg0laaZWFb5aua+SVvEY99+yZ870eaGi5YM6hl+gXJziVcw9Zlux6e0jjv0CvZPjQ4Nzf0Sf0EX1w2y/SsynpGvEOoPdoGr2J+oViWBCRS5TtXoLYrj2G9aDz6o8K/Nh7uP6LrJZDaJ3We3A3/FV9H8JaSc+s7TPpQa0vSVwwPZ/MPtS3kvjp1E/opQQ3fN9WAnpor2yt+9h3A+FMqB8IBAKBwD/hD5ii1rNS1BQBAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAaCAYAAACtv5zzAAABNklEQVR4XmNgGAXUAEpKSvxycnIh6OJUA/Ly8v9BGF2cWoCJphYADX5Dawtu0swCBQWFQyAaaPgfqlsANDwSaOh3EBtIvwJZICMjw4mujmwANPA+zEAg+zDIAkVFRTN0dWQBoGFb0PhzofGQhCwOAsD84QPEdUAfC6DLYQUgDVDD9gA17QLydwLZj0FiQH4lsloVFRVRoPhmIA4E4idE+RCo8AVQIx+yGNDgCKily5HFgfw/SGwnkBpkeQwAdIE80LAEdHGgL5yhFuxDFkc2EKhGm6AFuBTIyspKQS14jyyOrF5ZWVkWl36Qwq9A/AmIv4FoUVFRHiS5e0D8GSoPUvcTiHuhcsRZQC5ANhAYREq0sOAvjA2MO39aWLDI2NiYFcpeB8TH0dVQDIAu1wDiRJhFo2AEAgBACGL20C6QGgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAZCAYAAABHLbxYAAABu0lEQVR4Xu2VO0sDQRSFJYUWWhoDee2GxCaIYqGFdpZWWvhAECwEOwVtxEJLwcrOJr9AtLOxEQQrwUbFHyAq8U3ARkKM38QJjDfZ7K5BENwPLrtzzt3Myc5M0tISEPDPsSxriLqUukkymTykp0zl0+l0l/R/lVQqNcjEazpAWfpVbNtexd+LRCLt3I9x/5xIJAZkn28ymUyb1OrBZD3q2igo+oX0GJ9IzTcs0QYf8iD1RrgErfEYb0nNE+rN8GCRkMfS80K9MFW0VxDavNKZb8LUHaH5iCqx1/qk54cGQUPauzVF9um0Drpu6jXQNE7zKddcPB7vlr5fnIKqw6MDXZs640mlk2Hb1Ctks9lWjAUaXqPRaKf0m8EpqEJ7N6ZGjin9BTZNvQLirDJZ5l7pNYuHoHdCm9H6iql/g8DDNJSofen9FA9Bi6ZGhiWtj5i6IzTmqHdqVHp+cAn6IT2WfldqrsRisTgPvakNzjAkfS+4BD2QHuOC1DwTDoc7eHhZnVBqUfr1oP+KeqLuqTz1SL3wxubMPsa29bVy6of+TJ12r/9+jqhJqHOpNwsB+3kBO+pQSy8gICDgj/MJCquO9w+H+EUAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABLklEQVR4Xu2Tq0oFURSGx+ATiJa5XxCxGEbQIiImg88gnC6CwW6zCKJZi0ZBMBkVfAOrT2ASLCrK8VvD3rL5VZxy2vlgMWf+f91mz5koGgVVVWV5nh/+FkVRrGr+aInjeCpN05W6rme81rbtZJZl82HeD1h3g3WvuQ5dfFJ047xL4kRrOig6cMl3SZI0oWeT8d6sIZsloddB0baZXPfU8+DfE2eqd9hkW5ufE+p5yDknZ0t1M05tuuoKOceqRZxwivFOfKin8HibqlnXB5tO7KrXCwpfXYM19Xrhiv99foMDXFatdwOK55qmmVa9ezXWoCzLBfU8FC+R86J6Bye77ra4UM+D90gMVP/G/U2HTDric571Ovf76M9h7p+wyY4/D+KWeCKu+BoXNXfMmJAv6RROn+12BRMAAAAASUVORK5CYII=>