# **Institutional Audit Report: Quant Lab Phases 0-6 (Validation & Integration)**

**Prepared for:** Claude Code (Development Agent)

**Author:** Lead Quant & Product Manager (Lopez de Prado Framework)

**Status:** HOLD (Do not proceed to Phase 7\. Mandatory Patches Required)

## **Executive Summary**

Your structural implementation of Phase 6 (Agent 6\) is well-architected. The separation of concerns between cpcv.py, dsr.py, pbo.py, and target\_shuffling.py is exact.

However, mathematical validation in quantitative finance fails at the edge cases. An audit of the pipeline reveals three critical econometric vulnerabilities in Phase 6, plus one lingering vulnerability from Phase 5\. You must resolve these before we entrust Agent 7 with live capital.

## **1\. Outstanding Phase 5 Blocker**

### **1.1. XGBoost Sample Weight Normalization**

* **The Flaw:** As noted in the previous clearance, Sequential Bootstrapping uniqueness weights (![][image1]) are fractional values ![][image2]. If passed unscaled into gradient boosting algorithms like XGBoost, they trigger min\_child\_weight violations and cause vanishing gradients, effectively destroying the trees.  
* **The Directive:** You must explicitly scale the uniqueness weights before passing them to the classifiers.  
  * *Formula:* normalized\_weights \= u\_i \* (len(u\_i) / np.sum(u\_i))  
  * Ensure this is verified in tests/unit/phase5/test\_sample\_weight\_propagation.py.

## **2\. Phase 6 Mathematical Vulnerabilities (AFML Validation)**

### **2.1. Vulnerability 1: The PBO Single-Strategy Matrix Trap**

* **Location:** src/afml/validation/pbo.py  
* **The Flaw:** The Probability of Backtest Overfitting (PBO) measures how the *optimal* in-sample strategy performs out-of-sample relative to its peers. If your pipeline evaluates a single hypothesis at a time, calculating the logit rank of a ![][image3] matrix is mathematically meaningless (PBO will always evaluate to ![][image4] or crash).  
* **The AFML Refactoring Directive:** 1\. Agent 6 cannot calculate PBO on a single strategy in a vacuum.  
  2\. The PBO module must query the Alpha Registry for a **cohort** of related trials (e.g., all trials within the same algorithmic\_family).  
  3\. Construct a performance matrix ![][image5] of size ![][image6] (where ![][image7] is the number of CPCV paths and ![][image8] is the number of strategies in the cohort). Calculate the PBO on this multi-strategy matrix to find the true probability that the selected optimal strategy is overfit.

### **2.2. Vulnerability 2: DSR "Cold Start" Variance Collapse**

* **Location:** src/afml/validation/dsr.py  
* **The Flaw:** The Deflated Sharpe Ratio requires the variance of historical trials (![][image9]) to calculate ![][image10]. If the Alpha Registry only contains 1 or 2 trials (a "Cold Start"), the variance is zero or statistically invalid noise. This causes division-by-zero errors or artificially inflates the DSR, allowing unvalidated strategies to slip into production.  
* **The AFML Refactoring Directive:**  
  1. Implement a strict "Burn-In" threshold for DSR.  
  2. If the total number of historical trials ![][image11], the system must mathematically reject the DSR calculation.  
  3. *Action:* Return DSR \= 0.0, log a warning "Insufficient trials for DSR (K\<30). Auto-Quarantine.", and set is\_deployed \= False.

### **2.3. Vulnerability 3: Non-Contiguous CPCV Embargoing**

* **Location:** src/afml/validation/cpcv.py  
* **The Flaw:** In Combinatorially Purged Cross-Validation, a specific combination might select Group 1 and Group 3 as testing sets. This means the training set (Group 2\) sits *between* two testing sets. If your embargo logic only applies a blanket drop after the absolute max(t1) of the entire test set, it will fail to embargo the right-side boundary of Group 1, leaking path-dependency into Group 2\.  
* **The AFML Refactoring Directive:**  
  1. The embargo logic must iterate over *each distinct contiguous block* within the test set combination.  
  2. For every test block, you must find its local ![][image12] and drop the subsequent training observations that fall within the embargo window.

## **3\. Execution Orders**

1. Complete the Phase 5 weight normalization.  
2. Refactor pbo.py to ingest a multi-strategy cohort matrix from the Alpha Registry.  
3. Inject the ![][image13] "Cold Start" circuit breaker into dsr.py.  
4. Update cpcv.py to apply the embargo period to the right-boundary of *every* distinct test group in the combinatorial split.  
5. Write unit tests explicitly verifying the non-contiguous embargo logic and the ![][image11] DSR rejection.  
6. **Report back when completed.** You will then receive the final clearance to build Phase 7 (Agent 7: Probabilistic Execution Trader).

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAABKUlEQVR4XmNgGAVDHMjLy/fiwNHoamkPZGRkOIFYFV2cJAB0+g0gbgXizXJyciFocv+BeB2yGE6goKBwCEgxAjVcB7I7kaRAYiCDipDEcAJGkLdADJAmFRUVdpgE0NBMoNg3Y2NjVoRyuFwFuhgcgAxC5gO9uQIodgBZDAqYgOIv0AXBACiRDcS/YHygIUpQb3kiqyMIgBq2AvE8JP48JBcyycrKmoIYQC/tAorfhKnDAEAXLAAq2APjA9kfYAYBNTsgiTsC1YYCDdaBiWEAoKIHQPwUiD8CNXMA6SNA/BLIXgiSV1JS4gcFPFDsDbpeDAC0zQZkCIwPcgGQYoTxgXL+SF4mHwAN+QEKBlCMosuRBIAuOgU05CAwjKTQ5UgGwLBSQxcbBYQBAFOZRHPF/CdsAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAZCAYAAACGqvb0AAAC10lEQVR4Xu2YO2gUURSGl03QFAoqBJF9THZZjaZRCNpoI4iFhRqiFqIogogKotFGEa1EQbFIIQh2go1iYeODFHa2PsBKQsR3pYISjbj6HfYOjn/u7uyssyzIfnC42f+ce8/9Z3Yem0ymS5cuaRIEwRdiVPV209/fP29gYOACvX/xsVfz7aaHxhNsoE8TxWJxN/oD2xjxls8LtSYBvayxr1AobNaE0RHzGDrjGs/Cmb5hZ4e6Hfz9XmviYN5aDuAe5r629VI3z4KrabCFBUbD0Jp6OINjHv05sUm0J8ThqBZHPp9fypDlIOxK1TwTRoibRJX4TLwhJokXWlsPa1oqlRZHNTa6znT9mnOAr6JPRLVmSdU8xS/ZdKB6EjBzknWuq4722G1G9eOmE+s1F0dq5tn0scHBwfmqJ4WGU2xqg0c3g7PM0/eAyx3SXBypmafwo2qtwDpVNjXs0b3mqd3vcqc1F0do3u5NmjOaMm9HLqh9LVc0Cp3nwxpyQ1rk0X+4zfxF5Myf0FwcEfNbNWe4fo3NB3+uu4ah83xY3dDQ0ByP/sG3Bhs/aDrjXs3FEZonRjRnuH6NzdtjjcKvqreCNeTMV1Rno/d85ovunYAb7UrNxREx730MN2XeYKEjFC9RPSmsMYmRjarncrk8uSpneFVUp+999HHRzlYqlblRzUdonnGb5oymzUOW4kcqJoU1bmPwqOoGubvEJdGmOVhrRDND56Oaj4j57Zozkpi366+PCd8Z79hbFOMC5B6tawRzlrPGK9VDgtpL0zRxkdqnjOc8Nc+Ib5k6Gyc3Q3wKaveRMOzzjNQ1b95w1/+4TdTQ2npYbblcLqrusNfSYWquEDs1GULuJ0NW9SQkNp8GNH0XtPCDJQrzT6mWlI6Yt0cdjR+6yyYxzJ1SrRU6Yt6wFx2a31K9GZh3TbVW6Jh5g+t+WeD5adtu7H8F9L1skfnH+0aXLv8ZvwEpner8QbelTAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAAB70lEQVR4Xu2VzStEURjGrxHrWSiZZu65Mym6WZlINmMlijX2NkhKNhbWpITsLFDCRvkfJsvZyD9gpfGVkI8RxnPcc3R6Xfdr6kadXz3NPc/7nnPe98499xqGRqOJFcuyCoyxY+rHhWmaNvb/gKpSNAc1FtW4W843+Xy+ARNGfBNjAjXMedRSB3+Lx7LZbH8qlWqiCV8g2I2mRnFZ77FYrKCGc2hS1LNP4/hnV+GfUP9X/lBjVdzsdvy+Qa9QC4mXEF9TPU/CNibO5BP1JYidQZvU9wNz9vgv1u8TNZVJ/B2xpOp5ErYxDh6LIWiG+ul0uhX+Di4TNOYHahhXrn/UhHFJHfvitkgQeHO4g/NyjHPbBm9bzQkK1plQx6jnhteUyWS6RDyJ8aCa40vUxjjYcABzn6Fy1KYM5413qRq5XM5kzlmr4IY1hzpbkloa42DuLHRqRHj8OPhXOtz2h3coaltgYd6Gkloaw7w7aAN3tIdf03gQMO8I2qU+R9YGVWjMl6iNYU4vGlqXY/FtXFJzgoB1bpny4lCRtWHdIo35EqUx3hTO1KKL3wktU98Dfr5ebNtupAEO30M0VqAxV5A4xpxDfwVdCF2zAI8TNhuGpqkvsZyP7Ar1CQmx5wN0Dz1CUzSJA/+AehqNRqPRaP4Jn9M+nMzYwtb9AAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAABfUlEQVR4Xu2UTysFYRTGr8sGC+VPUzPT/GtsfAALSSkWoiwUxUrYIFkoH4DCVWJra4HN7S6spCyslLK040PorhSedzpXj+NVo7tS89Rp7vt7nvc9Z24zUyoV+ndK07RPs7zyfb9bsx8KgmAKdY9wexzHI2EY1lGTOqeFfIrcSxRF047jdOL3FepD5zLBONUm1mea2WQyqDvFllBPzBrGqz4U61XNbJJGFWb4ZwaseyX8zUB4TPg6cxa8GcksK6tNn5fpl0bDwneYs+BtSWbB4uVrhAdiUPg5cxa8I5PBULMW72+NcEiNOQtP2knTjbAeEn7InAVvTTLzFi93o1HDMO02cxbuaM5kcF1UVos+L5OtERqsGOZ5Xg9zpVbZu8sQL3G/Pi8T4LM2MOWBZmi+gdpjJnd0qdgEqs6soTKMakjvg0x6wyFheqBNYWWzxiDjWL+7rtvLuS/B6JDp9nG9xYaLJEm6OAPvOrR8WsCPwR9QFdQb6lFnChUq1Lw+AQoShJO0U1abAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAABX0lEQVR4Xu2TTysFYRTGL2WhJNEkzX9G2chiPoKF7OUTWCllo6zsJKzsrCVJ8gV8AGShJNsbC9b+lejG79T76nhn5mJjNU+d3vOe5znPnLnnTqNR418RBEF/HMcfOlyNRpqmE45+ydV8AfPuJEkWET38YNwJf2pNsyzrdQUFILzA/KCdMdwNQ4wb40OXLwDDPoQtYkOafN8fcDUCuLkoimaM8bzLF4Bok4YzzmlpIp8t0TTNeU3cuXwpEJ4z9ToxJsacy66Gh62wuEEz7Z7LlwLhu+d5PSa/J44tl+d5Fw+6lBzzbTHmdx61fFvQsGZzmYZ4U/em7MDkMm3lcr9Bmogpe6dxVTV3yLIU9yfjLX3HaNIYDJHfak7qbq0SiK/0ncZhY3xEvmDrLC429R2tr0TZq0lNPgSntiv1MAxHdL0ARC/EI/FEtOTvprgTlT8brXzucr7yJvuWr1Gjxi/xCSTvbwTkYt2KAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAACH0lEQVR4Xu2WzUtUYRSHncpVtGoxi/m4MxM5rZtQBMvWQfRFoBvBPiAK1+If4EL8D1wkLYKiFqELN7Zt3yZoFbQo2pQp+DGU+hx9r9x+3jv3zuAIxvvA4b7+znnPe877ofb0eDyerhMEwQ1sJ6uVSqULmuM4YO1N7At2B3uL/S2Xy/NWk8bugeONK3q2Wq1e4zuIfXfaU+wyCYb4LrokpzVHt2H9SdZ+ns/nz4ZapVIZt3rwfY3GHoCzSdAl0aypQzuBtqxat2HNlbhaDFfnA9X3oONXqrVo7INq3cbVsqW6YT4OpaK6HeewaoVC4bxL9iOqNxqNXtu9qBYHMb+40v2qG8H+e15XvRXhJtv66uO9X1UtEZLcczvxWn1ZYO40tqq6gb7BDXmseivCxrCP9taQTmlMJkjwyRLVarU+9bVD9GTcZsVepzQ4lSvM/Rxp0GzbcmpsIpzSqJv4Un2dQJ4Ncs7wfai+TqjX6+fI9yJsUP2JEDznJj1SXydQxEJbBWSkk8bam9AC8jyjsRE33mJ8S2NSOMObGlPRCPavYvY6j7ixZjimqSf8vBn1p2G/WZlzV3XD1XmQP43cUTTGLk8kNWE6Td5WPQ5il7BvqtvJs8Y7hjn1/QOT/2Dr2Cr2G1uz3cC24/52pMHC11WLQlE3VVPIMWB1FYvFi26zf2LvsRV8Uxp/YrDG2ID7bjxst8D+4U16cx6Px+Px/E/sArzwrcY5grKpAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAAvElEQVR4XmNgGKFAQUHBXV5e/j8J2AquGcj5Ki0tLYNkHkjsF0ihiooKO5r4VWQ+SOA3igBEDGwLFvHlcI6UlBSXnJycM5I8GODRfAXOAfrXHkgxIqQZGERFRXmgmj8hi4MA0KJH6GIoAKipB6q5BF0OLwCaLAjU9BeI3wO5TOjyeAHQGwEgW4GGbECXIwiAGi+DNCsqKuqhyxEEuEKZKECyZqA/44F+zAVqWgTTDBTLBOIE9BQ2CkYBSQAAwn4+E4xkh3gAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAaCAYAAACHD21cAAABAklEQVR4XmNgGJmAWUFBYYK8vPwFOTm5MiC7AMg+CqQjgPR9dMVwAFQwEajgP5DJgiTMBBIDyi1EEkMARUVFcagCCXQ5oO07gFgJXRwMgBLlQPwIXRwEgAbOQheDA6DkOqiNjehyQAON0cXgACiZCtIIwkB2LtAADXQ1OAFQ0xOYZiT8EV0dTiAlJSUCtDUMSbMhuhq8AKhhNVRjMrocKO44gBI/0MVBAChnANKINXCAkg4gSXRxEADKJeCSAzlnP0hSWlpaGIscyLZ6dHEwAEr+Akq6AuljQPwKaMspIP0diG+oqKiwo6uHA6DCACiTEcguBOKZsrKyfiA+krJRMAwBACYvQ4PFRGE4AAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAaCAYAAADi4p8jAAAC7klEQVR4Xu1WTYhNYRi+mfGXBZIu7s+5f6VuLHQtLKRJQlmwnR35WVohkils1NgajUZImbK0EE1TMw1iq/wUIkkUC//TteB53PdM3zy+c++cm2vBeertnPM873nf9znnOz+pVIIE/z5yudzaIAgeMlTrJKznM8QP1aaBCRqVSmWu5Ey4OqiuUKNBJ/U3lEqlhTwnn8+PFwqFHdg/gHiHeIR44Oa6PTzxBnHOzc9ms6upuZwP3RhyBRMxwCnEIk2o1WqzrUkd+nZXa2UQxm4g9rkc6lyxegMuzz6o32Oz3K9Wq3MYMFIB95Y8aq0P82dq8Bfs5D7lCTZBw0vKE80M2rDvlSfYr1gspj38edM2e7RXdmG28Ti2QcSQ8gT4o1i2S5UnmhnEecO4aDeVJwJZniHAP0HU0+n0ApG6yXNO1NxLoh2Dj5XnEgM/oXyIFgafRw2Au1vwcL02x0nVwA2Zdizk2jH4xcPXYbKqfIgWBs+wLga/hu0e1RXIG7Q5NqoGbpKa+wKMa/CpFV8TcjB2FkXmu3mKZgYJ1PtodacCz1egeYSTM4oYQf972L4IGi+Y5ZofyyCKXWQytvuN4pofnpbkQSuDITBMBvXuhiZUJyK0LnLlcjknfDyDSDxhDU7b8UHfW04xU4MElxfqjkUNZf0nfbx+aohYBvlatkJcFtcRtzXHB59BPEvreMeUJ6Bt9Q0Ffhd5bI+rZsb7lY9rcKUV4h9DHYOv0hwfIgwe5h+M8gS0I6h/R3lwl81gj0fjhT+kfCyDhBlkk3mqRcFjkM/uZ8QI910B3EDUQGZiXHnMssxm6rW83Ygx7rdlEE0uKN8MapB3D3VeY7slaPwM8+/jFuK7DTn1H0uA/2oX5APiE+IbZtjk5uD4ZdBYWf2ocRXULPKxDdIcYrHyzeAxuJONuc9PDA3zc4NHYIObFweZTGYJavSh1mDKzBGxDbYDNfg3kRj8E/gvDAaNX6tR1ToJ9rNvdmcNJkiQoG38BDaOGNvfoEGKAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAaCAYAAABM1ImiAAAEr0lEQVR4Xu1YW2gWRxT+ja2XtiAWYmxukz8JRaKgIUjVB0EQpLSKoiA+iCkI9aEPilQQvKAPgooRBa8PkbwoUiq0innw0noDRVFEqrYQqhRv1Rg14i2a+H3ZM3Jydv/wJyvqH/eDj935zpkzs3NmZ2Y3lUqQIEGCvgnn3GLwcmlp6UZr+xjBsRCetraskE6nR6NyRzbU9VBerMu5grKyshHo+zlwNzgfE2klrpfAGWC99kW53Y6BJutWVVUNMHV6lwggr7i4uBId3CEN7CsvLy8lgXIkaiK01bTpSi5HE4F+XwfvGO0inw9jME/rBLSlYmsoKCj4vLKycmBJSckoaHUyXhe1f5xEEP0Q4C7YXlFRMcwaCdiemnLOJQJ9Xg7ut3pNTc2n0F9anYD+NwccE3NIhI2J6NBjFisRkmEGvWRtHuyQKediIv5jMqxO0GY1wg+21Qlv4/KutN4nApV/ZUC8ftO9huRU6A7g/oK/l3IoEZxZWOaKfFnfezAu9MFW9ygqKirGkvg9bvOsLS5k4JrBamtDv6ZZjcgmEUaLlYh7DIgBGKq0LeB17adhE4HyI8ZAMn+X8inpKGP0NxrbqtL1xd4CXgFvgf8i1gJl/gTaefAs6p7B9RxF+MyRuCepK/8QXLAps/1n8J2FCfGl9bEQ/zaro92FtOG6SesuZiLYWIhopNb6eriINwLaCfAlHvKY13C/hhp432uIO0bamKn8Voi2TMpzpQ/feB+Cb5P4/eU13Ldqn0zgCcdFnITQxiDrS0CfLT63wUMoH5f7q7hfav0JFyMRedJYl6yj3KTLdrNy0Yn4BXyUUsuKephvlWtnm7CtVVoI8HkAbrE66v0g9X/E9Ya1ZwO+kaj7k/TtlrUTLtionxcWFn7mNb5FUidyP3W9TQQ6VCOBuwRAeacp/2zKoURgYPZCv6k1LgGMD9s4rUub67SWChI0AWwEn9AH9bcZn04gXoPE+MPaegIXLIOM81WEjfrJDHqmfaN3iUDFgwyKB5tibR6wH47QQolwwUdSl33FBR9KHdgMxxqdD/MmERjwJaIdUD4t0Lf7sgZs34EvWMfaLCR2lzfcA8+9ypnvCg/GRt3JUXqmdl2MRLSCbfn5+V9YG8HTRFSjLjoRe8BrRusuEet5j2Xva/YBbEzJxi4+rT4R+gQGPQ3eZx8Yp7tJRDCuMx9eHtD3g3VWJxhbL0sEy9L30JgQLkYiMq3V/aAvEnutNXIQIrR9zqy3fmkCx2tdtA3GZ7O3y0zljNwlPlN55V4F/ZX/iMKXbj79UiqBFhI7NHCy8T+zOoH2F0TVcTKxtM3JhJL7niXCBSeZ0AlCSJ12ztLOU4yFCx9fOUP5dU42c0/A9aELjsb/y3WDC9Z+nuXviHaN9TmwUu4An4LV6oTUhESMFDvr8Zh7VNqtF51tNNt/PwTqHuPvCRf0pwn8E3yFPv5mfaH9A9tjFxzHyefgE+3jgv51TmBcW4ytZ4mICxfxRrwN4OFGICklvpzpl0tP4E98iD2cbxrYwOXQ+mULPHu6LPh+2QpOMra+kYhcR5KIDwRJIj4QvJdEgEewEa60to8RHAvhu01EggQJEiTok3gN/a3S7QBXwesAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAaCAYAAADsS+FMAAADIklEQVR4Xu2WS2xMURjHi3pGJBaDTDpzZ9phUYRoIvGIjQZpxMoj8YqNLhobVkQQEkI00SAICdEF0VRjJejC0oKwblTUI0SFoC3RKn5f5zvp8ZmOue2ksbj/5J9zz/97nO+cc++5p6QkQoQIEYaJ8vLyOUEQ/LLENNb50F9i7I1eilFDMplcydgXYQusicfjU6yPIJVK7cbeTHvV2v6JTCYzkeAjMlES3KusrJxgfdTWREHrrW00UFVVNZ4avsFL8IBuSpv1S6fTK9Cfwzp4Gt6n7lnWLy+Y5EsZoKysLGP0avROKcbXi4hSK+RAKTX0wBexWGyqCDw3S71M9Lhzot8q2mDYgNZmtbwgYUoCZEGMaQz6J/SNRi8KyHuI/G+tblFRUTFD6tMaq0WjPa/9W86P/k87cabWYLW8IOEVTbzdafTb4XXfrxjgNV4cZHf1trWFAfG9WvMyT5MF6/b9sO8UPZFIrPP1IYFzhwSwAwnps5o7ZOWt30hB3tWM8wNes7aw0InbT0K0d0bbrHqdrw8Jl1g/l0ewT/rs4gLrGxZ6OLezuMfIN9Paw4DdnUeuLq1tlW/jrJus83jl63Lgi057yteHhFsMWCN9Apdrv8f6FgIWIEbsUfiUXLXWXgyQ+7FOcq5Kcr5Jza99P+wbVK/39Zzw7hkdvq5a4QePB+Keaew4aysW3FkAn5ToH0n7b3w/XvZNotPu9fWcwKlJnO0Bg/ZBdAZd6+thQHw97CLHNmsLA/ndk+dcyrsv8PZN08nLRPeJpv3ewcgBbZf6rPH1nMDxPeyn4Om+TvAJTd7q62EhecnRKUXJd23thYAcd6UW2gtOY/PibjGS+ut3/cHIgXkcFq2gsXXVGqwukIXQAS5b2zAgF6ct8As8aY35QH03tI6lTuN5v2gsxB2nybNdjEDfcF/7Czh8hd3wc5C95vYxaMqzy41PEsmufoT9XviIQb4W/UQXWVsusLPzg+ylSs6jB/p80/qhpeF32AgfwjOMM8n6/XdIZm+Rf1yS8kH+cvAgMWdZnNnW7sDkF+KzB9+t1hYhQoQIESJEiDBa+A23OgWZObUGvwAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAaCAYAAAAKYioIAAADmElEQVR4Xu2XS2hTQRSG04ric6PEatLmtkk02o1gN4ovxAciClLwUbAqBRWFunanoIiCulAEXyB0U13UhSgiig+0ID4WgrUFpYhVfGupT2qlfqeZqdPTmyYpiAbzw8+98/9nzp05M3duEgjkkUce/wM8zzsYCoVGaj1bRCKR2uLi4klaz0lQlM7S0tL1Wh8syNcMb2s9p8DqhpnENa1b4B2gaKu1PhDYNQvo1631nAITaOdSoHULmSCs03o6UJy19Dui9ZzBQCsbjUYjpjA12kuHioqKofKKhsPhcdrzBVt3hBscj8eDri+g2tHy8vJhWjcYgr+srKxssTayBTkSDP6L1i3wTsLHWs8UUlRew2Na7wMGMY3AbyZ4O9fd8IFZkYcSw4SXc99ktJ9wokpTKBOBt2AHvEuuJdakPRPeR7tjWCW63KM3Sj83nudVez7nC9oj2AK/wu/wMbyp49LBS86jVev9UFRUNIrAz7DL1WlflEEkEokxVmMCaySxirviatwfkrbervK5VHEv/AYoz+U5O7UuQN8gOUpKSkLayxTSX88hJQh8CS+7GoPYpxPQnqo1DQoQlxhWfqP20E/ADrx1XCu1L0BvFl/rAvRnGTw/rDUX9G9Ll6MXXnL1GlyNQezRCTj4JmtNQBGHE38U7yrsNIXZ5Bcnnl8OC7znxK3QuiBVX4oxluedwmvjOkv7LjxzLGjdF7IS8LSr0XmXTsAWjimtgEmcczUbg77ZieuFlzwjpHArtSdAvwFrtS4whWnRugXe8QwKI+dgZoWRSsN6paUtjH3n3TPBfFV6CsO1MhgMjrYe7aUMfBHeGe7buY9az4lpQN+hdYHklZ1p29yvUn4mhenW80oJAt/As64WyeBV4v6wKULvr1DaF8wEthmvVHRz9jQ7cT2vnG1bEL6Fvpe0jrbQ5O0pJnF7uRS6MV7mhXmi9T5gdecS9B6+NfwRSH5+P8J3XrJg8imugZ9MrGjt8Lrk4DrdxLTCLjn87M4yPwe2esncr2GHfbbJIc8Qr8nqxutXMIH3ewz3KMwUH18KM1vrLiQ3MdVa/yOQX5RSZPcTHYvFxrsx2SBVYeQHJgWZF0jxd0EKw6LM0boDWfRXA/xQ/bfB4OvgfK2ngxTGFM4X+I34E7SeM5BzhEmc13o6yKGe6jXBm5FqJ+YUmMR++EHrgwW5nvr9B8xFFDCZejm/tJEt2C1V8iHQeh555JHH38AvidUjvMPnal4AAAAASUVORK5CYII=>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAaCAYAAADsS+FMAAADNUlEQVR4Xu2WS0hUURjHrexJBFFGjM7cOzq1sKLAVc9FSYREqx4UEUG4kTa56g21qgiKioqCHraIzKJFi8pFtAoqatVCKNKKQMPooUWK2e+b+S4evpzxziiCcf/w597z/x7nO+eee84pKooQIUKEAlFeXj7f87x+S0zjAx/aS429wUkxakgkEmvo+xK8C2tisdg06yPwfX8P9iae161tSKRSqckEH5WBkuBRZWXlJOujtkYK2mhtFvg9g6etPhxUVVVNpIZf8DI8pB+lxfolk8lV6O9gHTwDH1PLXOuXEwzyvXRQVlaWMno1eocU4+phQOw1YnsoZp215Yli8nTDtpKSkuki8N4k9ZL7WOBEu1m0gbC01mK1nCChLwEyIcY0Dv0r+majhwa/YUIGQhdbaU6w9jCoqKiYI/VpjdWi8byg7XuBH+0/duCyQq2WEyS8qol3BBrtN/Cm6zccyBclXz1sh/utPV+Qo0drXu5oMmFdrh/2WtHj8fgGV88KnFslgC8QlzazuVNm3vqNBOhnl5dZuiesLR/owO0vIVq70bapXufqWREk1t/lBeyVNpvRYus7kqCPTunL6tnA112I/w+tba1rY6+bquP44Oqy4YvO85SrZ0UwGbBG2gSu0Ha39R0uyD2TvAdhW7ajMQyIf6mDXKCS7G9S80fXD/sm1U+6+qBw7hmtrq5a+I0nBPiyMXJ+h68LOZ1cBHsBfEWzWDRtf3L9WOxbROe519UHBU6N4mw3GC+zhGXm17t6ISDPbXifPlZaWxjIcU/8ed+5L3A3mqGDl4HuE03bPQORaW23+gx9vOP4GfbJ8nV1go9r8mZXzwfErpZ4eMPa8gG1PZRaeF4MNF1lUp/o6aM/aA9EpsdxRDTZU1x9UIijn+W2qAORDq5YWy5Q3BNiOtkTZltbIaC+W1rHskDj/YBOxINAk3c7GVKH1f4BDj9hF/zmZa65vXTqO3a58UmiDvgF9jnhow6+7CIvc6l6C5/q+x3rh5aEv2EDfA7PMqwp1m/MQ045eJgBnmNy5ll7AAa/BJ96fLdb26hDjkz5RcKwtLR0lo3/r6DLuD8sbXyECBEiRIgwtvAXqn4avRQiwn4AAAAASUVORK5CYII=>