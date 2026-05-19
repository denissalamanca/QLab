# **Institutional Audit Report: Quant Lab Phase 3 (Microstructure Features)**

**Prepared for:** Claude Code (Development Agent)

**Author:** Lead Quant & Product Manager (Lopez de Prado Framework)

**Status:** Phase 3 Code Audit & Mandatory Refactoring Directives

## **Executive Summary**

Your implementation of the Phase 3 feature engineering pipeline (src/afml/features/) demonstrates excellent software architecture. The FeatureGenerator base class and the \_apply\_shift causality enforcement mechanism are well-structured.

However, a strict mathematical audit of the underlying microstructure algorithms reveals **four critical econometric vulnerabilities**. You have implemented theoretical formulas via element-wise approximations rather than robust statistical estimators, and you have ignored the stochastic time nature of the Information Bars generated in Phase 1\.

Do not proceed to Phase 4\. You are directed to immediately refactor Phase 3 to patch the following mathematical flaws.

## **1\. Critical Mathematical Vulnerabilities**

### **1.1. Vulnerability 1: Kyle's Lambda & Amihud Instability (Element-wise vs Regression)**

* **Location:** src/afml/features/kyle.py & src/afml/features/amihud.py  
* **The Flaw:** Theoretical definitions state Kyle's Lambda is price impact over order flow (![][image1]) and Amihud is absolute return over volume (![][image2]). If you implement these as simple, instantaneous element-wise divisions on a bar-by-bar basis, you will generate massive, unstable outliers (and inf/NaN values when volume is near zero).  
* **The AFML Refactoring Directive:** 1\. These metrics **must** be estimated via rolling Ordinary Least Squares (OLS) regressions over a causal window (e.g., ![][image3] bars).  
  2\. For Kyle's Lambda, calculate the rolling slope (![][image4]) of ![][image5] regressed on signed volume ![][image6].  
  3\. You must use numpy.lib.stride\_tricks.sliding\_window\_view or numba to ensure these rolling regressions execute efficiently without Python for loops.

### **1.2. Vulnerability 2: Corwin-Schultz Brownian Motion Violation on Info Bars**

* **Location:** src/afml/features/corwin\_schultz.py  
* **The Flaw:** The Corwin-Schultz Bid-Ask spread estimator inherently assumes that variance scales linearly with time (Brownian motion). By passing Phase 1 Information Bars (where the ![][image7] between bars is stochastic) into a static rolling(2) window, the fundamental assumption that "2 periods have twice the variance of 1 period" is mathematically destroyed.  
* **The AFML Refactoring Directive:**  
  1. You cannot use a static rolling(2) index window for Corwin-Schultz on Information Bars.  
  2. You must calculate the exact chronological time elapsed (![][image7]) for the single bar and the combined two-bar window.  
  3. You must dynamically adjust the scaling constant (traditionally ![][image8]) in the C-S denominator to reflect the actual time variance ratio between the periods.

### **1.3. Vulnerability 3: The Stationarity "Drop" Trap**

* **Location:** src/afml/features/pipeline.py (Stationarity Enforcement)  
* **The Flaw:** If your pipeline evaluates a newly generated feature via ADF, finds a p-value ![][image9], and simply drops it, you are committing "feature starvation." This lazy dropping violates the DoD constraint to output a high-dimensional baseline (50+ features).  
* **The AFML Refactoring Directive:**  
  1. If a feature fails the ADF test, the pipeline **must not** drop it immediately.  
  2. The pipeline must route the non-stationary feature back through the Phase 1 **Fixed-Width Fractional Differencing (FFD)** module to compute its optimal ![][image10]\-value and enforce stationarity.  
  3. A feature may only be dropped if the FFD algorithm mathematically fails to achieve stationarity without destroying its correlation (which should be rare).

### **1.4. Vulnerability 4: Naive Entropy Discretization**

* **Location:** src/afml/features/shannon.py & src/afml/features/lempel\_ziv.py  
* **The Flaw:** Shannon Entropy and Lempel-Ziv complexity require discrete alphabets. If you are binning continuous returns into just binary states (e.g., ![][image11] and ![][image12]), you are destroying the volatility context (a 10-pip move looks identical to a 100-pip move).  
* **The AFML Refactoring Directive:**  
  1. Implement a rolling quantile-based discretization (e.g., encoding returns into quartiles or quintiles: ![][image13]).  
  2. Calculate the LZ complexity and Shannon Entropy on this higher-resolution sequence to capture volatility states, not just directional states.

## **2\. Execution Orders**

1. Do not proceed to Phase 4 (Feature Selection).  
2. Refactor the microstructure formulas to utilize rolling regressions (Kyle/Amihud) and time-adjusted variance (Corwin-Schultz).  
3. Upgrade the pipeline.py to utilize FFD for stationarity rescue rather than naive dropping.  
4. Ensure your unit tests in tests/unit/phase3/ explicitly test for inf/NaN generation in the regression outputs, and verify the FFD stationarity loop works.  
5. Report back when Phase 3 is fully mathematically hardened.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAaCAYAAADv/O9kAAAC3klEQVR4Xu2XO2gUURSGV/GFb9Blw75md11dXGxk8VEoNtoJviorLSwMCNqkiqCFVqIIaiFGtLSwjg9sBEVB8FX4gqCIWkQQMYkhShL/A+fA9Xfv7MwuBFnmg0Nm///cc+6duXN3k0olJCTUarUluVxuBeszQRAEv1mbEYrFYkWaI6bZ84HcN5IfFqVS6SCPYxqNxlzk7Wdd4HoSjjfh8yKDQVfbHDzHxpXL5Yx8rtfr87CQTdAGRM9mswt5kAtyd/ly0un0YjyUhvUoFArrzKtUKsugPUaMoca2tnYrBg9nMplF2mCQ/TBsUqwL6r1l3ZDJwv/FOoOcKaklN8HRHiKeu3mxwJ0rokCfXNsicGc3cJ6PCAv/xLoB7wgW85F1Bjm3pRae7GWVZKcN4XPPX4lRwR3Po8CEfUaDnWELYdB4t+ZfYE+2vHp72DPUX8t6E2ShXzT/DOI7J8QCBS4ivpImxaMu/Lzm723iHW5VB/4T1nwg95T2GseO3Mp+LGTRcoCQ9lQb3HR1Rt9Peffe42DbKO8fYgsW3KvjR3iMC3K341BbyXoYWneA9VigwFn57mYdzIL3QppgcjfYNOBfkhx5stVqdb5sbc7xITcbY0dZ9yGnOfK/IT5oz17OiQwKjLFmwNund3eSPQPeK8nBpFax1wpM/IAshHUfyP2Mh3ANf/t1Xu2d5rJNUegE6y7aYBqn/hr2BPNZjwLGTWHxm1lnsJOWIvcZcheY1klfGXwaxZaz7oKc19IAN+g6e0InE0Dvd6w1A73vI/elq1nfVvP/Bww4qYPv4fouit/B9S3EoBvi+RaXz+dXqzfEXivkNwIOw4B1BrUfINY30fukt9682ex7scXEjHM69griJ+IHYkSvJ+McNgF9fbqgTg/8Uaf+OLRj5uPzcfXFs5xht8Z/SxByqHYt8o8FXq0drHc9eNqHUnHey25A/y/w/i7oWrDFj2Lhj1hPSEhI6JQ/FwAMbuJhxwoAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAACxklEQVR4Xu2WOWhUURSGQ8QNRRAZAmYyLzMGizQiUwha2VmIS+GCIAjRqCnEys5CRBBR7NRGUNDWwoAWdoIBEUkXm4BINC6F4m5iEv0Oc6+c/L43SzSDwnxweHO/c7dz38yb19bWosW/RZIkfer+BOYbU5dGd3f3hnw+v1T9X4ONHFQXIfejSnwh7qaMea5Ooagl9HsX23zelzL/Jz8m9PP5iY6OjmXa5xeFQuGQOkd7T0/PijDRDJ8X22SdnZ152o/NM/6EH4B74dtpMGY3/d5619XVtRo3GtY6j2r3eYMD2ULuOzFoe9H8LFikX52HyfaExS5ojrFXLOcd7XHfToM+U4zdpB53Max1WnMGflhdJrUKY7JLthj9tqbkBudY2Kg6g0M8bvNxva45g9wHdZkwyWF1EW53LhT1UHP4nWk53EvfVorF4jq+dmvUG8xVDnfst8Jxt7gsUJ8JhR1RF2GhXeEEz2oOP2I526j4V76tJClf6Ug8SGLCew5iG27Ku5qw6aPqInHzxBBxL1y/EV+JjdrfwL9W57Hx6jxhPf16j+VyueXe1YRBA+oiGYsMmONA9nofIfdGXcSesHb66j2yZjufH83qUC+NFmYnh5uisNveR6oVRq6vXC4vVO/xa/JTOJY08sDwZBVWKpUKYZEh72lvDv6J95GswvBFYka94gtLKner/geGJ6sw7siBsMgZ8afMc73vfaRKYSeJEfVKLCypPHXXa75usgrDjxOT+tqCGw4L37S2vYUQq1w+q7Bp66teCeva/E811xBaWFJ54n0m3hMfk8o74bmYtxdX2neISeKqbiBJKay3t3cRv5f96tNg/A0rTH3DaGH1YA8Ae0tgs5eJlT6XVhh9d+idz4K+25nzmvqGmUth1dDCrPCkxn/XvNCEwvpxz7xrCvNdGO1p/jrWetcUmlDYA99u0aLF/81PviXoYtSkz3MAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAaCAYAAAAAPoRaAAACjUlEQVR4Xu2WT6iMURjGr+vSTZTUhDEz38wYRqMszIqkrCgpKbGQInRlQZKFEgtRd8mSkhULRaKoKxspsrGwsuNK+ZcuV/lzXb935pzvvt57BnczY3Keejrf+7zP+fOe831npqcnIiIi4h9EkiSX4bpisTi3Xq/PKJfLS4lvkerTvlwutwT9CL5HtFt0rmtBIeOKP6SlwJOkphnfCHxGbrPzygZ1NyjiK3wDn1LYACdfCHjuwT0+xtdPPFwoFDZpX9dBCreahvsM5KQXap0NuIR2VWut0GsFgz4GW2DFduBPxbtvfNzqnPpRtymrbK6BUqlUxXRHXhGbQ3ugnl/AUZ1vF6R4OVWK3MZad1ar1Tkm/y5UPNoB0emz1eYaIHFFWkwjDH7I6zyv1wPy/DE0gQZj7cDzYSq0Y4Qg8/rvnBt9HvETOKjzwokeTbCevS53zOYkudw/u87p7UnuIdo1H8vkoQk6gUqlknFF3ZWYdiy0NmrYJ7q8/jaXQn47dWd5rYi/0+mg1/L5/OLQBB1Cryv+mwRJ87OYtDbe3v3Ot9vmUpA8C1/5mKJPu8HSN4GBbqCf8HEItVptJr5lU6EdwwLPQKgwV1RDpx1q4TkuOuuu21wKDI8xXFfxpMGIP3E5rtCaBQvd5Rf1t7RjWMi6Ar7prv97CWgPS5zNZmdpUzJxqK1/yTDchPdV/MvllslkZhc69GeBedew6Wu1xtpOwVEuv4rShvCe8TG5ReKBq73WEphew2H4Re4ATnF70vwrKRtx3vrbCYq6yBre0t6WQ6F9bj0Cci/hZzgIx/Rm/BYU2w83sMvzvSY7i1ZUto4haX6/F+BG+zuvIBfhSniOCzprkxERERERERH/B34CUiTckhQmizcAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAaCAYAAABsONZfAAABF0lEQVR4XmNgGCFAQUHBQF5efrmsrKytnJxcLpD9CyjMhK4OBQAVvUDjdwFxFLIYCgDa4gCkmNHEGoCapiGLoQCggkPoYkANJ4Di6ejiYACUXAeUbEQWA/ppB1D8B7IYCgBKvgUqsgbSU4D4DBAvAeLfwADxQ1cLB0AFu0G0kpKSGlChqbi4ODdQ7D/Q9pnoauEAFMToYkCbD4I0oouDAVBCU0VFhR1dHGjLJXyalqGLgQBIAz5NT9HFQACq6Qq6OEgiCWoaPKmoq6vzAsX2AnETklIEAErcBwUrkL4AxF+A+DswAGYAsRa6WjgAKvoLVCAIYktLSwsrKiqaoavBAMAQmoguRhAAbQpCF8MLgH4xQRcbBUQCAJSiO5l3qsCGAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAABzUlEQVR4Xu2UvUvDUBTFq6KCiOJHhjYlbUpAiGMWwcHBTRCK4CYIgn+AIILg4uIqCg46uos4OKmTrrqJoIOLoIMoCA7V+nFuuQ/CsU3axYD2BxfSc+677+T1talUk7+K53k92Wy2n/XEQBgvl8uVUV/s1cL0R1TZcZxxXlc3GPCOOpRhGBSwH4UJwTpOvlP0fD4/wl5dYPFDJpPp0iEH7EehoT5ZF9TbZD0WLNqRQPp8oYP2uK8a6JvSF9lgD7Sot8BGJIVCwcHCN/MZA4pRb87IKejGRfZwDUbFs217gL1IsKiEWiRNQsnd8sM6g9MdlPCoc/YQclXDLrMXCRalsfES69CmNdiH67pD7Bvgb5kXCJWEvENNyEXnNbFg83XLsrpZB60Yeqmb7LJpgHclPRJcAgRB0M49DYOBT6yFkGCVt2fDEOeHkFmPLP4Ap7SGS97LehgMupFNcS+22RPqDSW/vtg+BOpD0wvrDHrmdONSFc/VwNfshUHPCepe55xh72HuqQDzWZsaqbSsRYgxDJ7H51vVj6DN4k5ZvI9B+1ZYTxQEevV9v4P1JGlDqGN5qPFL/33wVU+af3s8z7CfCPK14aROUfvsNWnyr/gGeSydky/pKiIAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAaCAYAAABctMd+AAABXklEQVR4Xu2TuUoEQRCGV8QDEw1EcK4eZiaQAaPBzEgDESPxiMwNfQEFQyOfxMjc6w3EzBdQEAMvRERWv3KnoS0Rl5kBQfaHorvrr/mqqd1utXr6U3meN2KMeVfRJpbFj6JoXXmvfDOuOT8qSZJRPlpyAJOuL0DinEYbRVEMuF7XsvA8zwdtLgzDGXI7bl0lAbkWODdck3MQBBnnG11XSUAT05l3G/A057M4jod1XWUBPirHcwd4TPu1lGXZkJ299mqJMayazj/jE86P6emaygL4wig2WU/KBtu6xpW8AZ37JoAxoCt7LkdzKw2wFtxaK7y5rkZH0SWQQ5XbL29/4OatqN/9FU7RRZqmEzovKuEa0Mc4cvJvciFi6ovr+36A+UQ8EI/EM7FoffbHZU78e6kFsucypKk0cXONyTT1crW48SyxJXuanCq7ngCuAJ+XvV0bFQ8sZenX+Z7+mT4AxsNk195+nm0AAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAABT0lEQVR4Xu2TrUsFQRTFH5gEFQ1reO73Lm4XwWYzWCzWF/wfLIKCxQ+wmqwGg1EwiU0QLBrNBoOgaFKLz3Ofd+ByZsMKG/cHw+6cc2bu3B221+toiyAIJsqyDFhvjTzP4yRJvjGG7LWGFriUIlmWLbNfh2T/dSiEX6uqmtSF1+zXIdk4jp9YrwXhYymg7zda6IpzFv28wzRNN9jziKKoj/CXm+NkK00+A3IDyYRhOMeeB4KfGNukjYrglEtWV28fBQ6Svzu8lXeMXVhjnB1RFMUsgjusQ1t1hdDpIvuCen3WPRA8wn8xxbqALu60mwv2BPFY80CLhwi+s25x3bCOeyjrdA+EPlBohnULMg/azTnpZxiPbg5/y/pOnEZoj3UGmXXt5of0Z+xxYuan1nfimy5uPPBfzJv19xibuPg1PF/s3q2CzRdwN+Osd3R0NOMX3ahoGcnSlQ0AAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAaCAYAAABGiCfwAAABVklEQVR4Xu2TPU7DQBCFg2gCJaX/jYREmZYC5BtwAg5AyQ224RbQpaKnQ6KFpOAE1BSUCCEqeBNm0fI8DlksU+WTRvG+mZ2343hHo55UVdWsErwvGjRx3LQreG80ZVl+sDYIYoQTO9YH4d+mWjZRlmU7rPXCmgraEw5xVhTFCZ7f9TUfc100PFmSJNvQrvwaRgeIF8RrWBcNGwloesHTYsJd0f78WsWImwrQLg19U7Q8zw9J/+K3y2c07KSu6yP938acW6BJx7pnWY5Br3PzcPp6bjXEsLFqWOsCPabaZ8K5b8REisSUc+YpO9Aez6y30MIfjfWULtQs8OXtiUmaphnnTGQqbd54bRUjAfse8cnfB9IG9u4H6zZ+OjGMMLrGZFuhhr2n4dpEDNRwMSXnLfRwM/zeIeaIB8Qb15kE0znOMai78fUcXGsSVdwXvQatK7BmzWB8ArRNeA5bESTWAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAZCAYAAABggz2wAAAC30lEQVR4Xu2WS2gUQRCG10RFBREURffV+4LV9aQrKCII4lFEPCjEoydBFA0SDZKDCCIB8WDQq4r4ungRfB30pAge9GoUHwR8EHOQEHGziV/t9mCnMjM7u+rJ+eGnu+uvqq7qne2ZRCJGjBj/LYwxp3K53HPGOuzF1K19goD/Q/gdjmSz2Ytax3YY3odnYBW/g7JGmqt9/ynS6fRCNq/R6PFisbiC+SR8pv38QMED+F7LZDJF4vcyH4VrXB/WJ+G04hfXZxZKpdJyEua0vVPQZEoaY1zq2Ui/EluNJta6vn6QopVpjtjIccAzMD+BbYh8dxj3Y+py/P2B4244QfBGrXUCY0/bx/4a9mu7CwrfEhArjT7w1vj1uXpbsL/EOHyRiHJCAZCiAoq9C2uJkP8R+kvj8wjqnMyPuXrHkJOXDTm5Q1prBV2UB36Rm2JnXK01i24b+1YLOifzXmq7ITXC03CKvJfdmMhIJpOLSPAR9hcKhSVaD4IuyrFft9pmrQnK5fJiqw9rTeekqSM0+kRqtPpjvz3bhZx0j2le94Na1NBFebC/gGjrtWYhl47czm+0EJTTQyqVWmZ9nmqtbVDovbDNPAQVhe222Hk6slrzgP4ZvvOx++Z00GWa7+vJarU6T4stYU9qDA5XKpX5WveDaV5os4oyzcdrVNtd2MMc13a3Ucajdr1P+TT2hatceyjY8BIBP3hp79RaK0gBXlEO5PGXV9gu14jtgrvmYNM+sY1GiT1v5+dgPZ/PG0dfZ31u/Y4KAQ1WCLgKH8FtWo8CubiI/USuHZ4t1/zCmZYvJsfVu2VnXE66UWkITHHoG2RNrk2sf7o+rAclLuxv0YB8LJiIXy5RQb4JOAKH4Bi592gfbB8S6r0qn3429gp1vWKsax9ZS/PwG/wK3+O7VfnMhBQgiRkLWvsTsPECcm5nPCuvDq2Hgbgq7KOuHhpPal1gfQaE7eaPESNGjBgx/gJ+Aakj5fCK6m0GAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAaCAYAAABhJqYYAAAA10lEQVR4XmNgGEZATk5ug7y8/H8o/ocujw4YZWRkhKCKr6FLYgCYYqAtMehyGEBBQcEfpFhRUVEeXQ4OgAokofRZIF6ELg8GQOt8gZJHgfgvEBtB6RR0dSxAwbVAvAUmAGT/AjkBWRFM4gW6BJD/E10MJvEfXQKbGLLEQ3QxUMQgi4EAE1TxYpiAlpYWG1SsCIg1UcIZKjEPxgeG7y0g/zuQyQykX8rKytrCFQMlHUCCQBMeAel3UAP+yUM8bgRXiAQYgYpdgFHMCeKAohoYc2boikYB7QEAk7ZAvN8rxWcAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAZCAYAAABQDyyRAAABdklEQVR4Xu2UPUvDUBSGW0QUQURR1JJvghkdgoP+A11EB8HN/yA4uLiIg7gIUhDRUXd/gruDf8BFqCA41WpRUZ9b2xKONx9GCg554CXhPefcc3JvklKpoKCHWJa1Y9v2K/pA22EY9sucnuE4ziBN313XnfR9f4L7N3Qj81pUKpVxpvWknxcaTauGqnHHY/1RvAaaj+a2YNplAi9oQcbyoLYbfWr8a7Qv/S4MMkVCvb1VfTKeFdU8ZoALna+FxE1UY+u2ZCyNhAFOdH4snOEABbdoN3qeacQNwA4f6/wslClcQU1UlUFJ3AC/3gEJxedZFkgY4FTnJ+J53ghFD+iOz3VIxnWQ+6RrxBFc4jelr4XkQ5IbvIRrMpYGtevtAcpRH++R2EbU+wFPPEPiGYlXXBdlPAtBEAxTe49WOx73S2oowzDGorldTNOcs79/nbMylhfWe0ZVdITqsU+vJlVnw3S+jP0F9RlzhHusfaDeJRkvKCj4F3wB38ZiwRp8LmgAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAZCAYAAABQDyyRAAABb0lEQVR4Xu2UPUvDUBSGq+DHIqIoYoj5hkDAKZPgDxDH4uIi4uLg7OIiLg7iKIKKrg5u/gHdxcGhu5uDiINWpH4+F1qJh3vTRig45IEX0vc9957Tk9JKpaSkiziOs+m6bgN9oo00TftkTdfwPG+Qpu++709EUTTO8xu6kXWFURdLT0KjSdVQNW55bGMEr45msrWFoPkWF9xLX6LWjb40/jXakX4utm1Pq1WiC5mZUM0NA5zqfC0UXqrGagCZtSNngCOd/wsKqugKHdI8knknmAbgFR7o/EqSJP2Eq4SPlmWNybwopgGMG+AXuqSCv6xbR84Axzr/BwaZpeADncmsCJx/1jViy+f4r9LXQuEJanBoTmbt4Mxic4CerI/3QLac9XIJw3CKQ3U2s8DHXpmbiON4iHN3qNryeJ5XQ/GaR7O1HdG8cB3dojWZm6D2Be2jPfRU6Nvr4JIVVJO+Cf6KB9jeNmd2gyAYlnlJScm/4Bu782GUMSSAAgAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF0AAAAaCAYAAADVLFAXAAAD7klEQVR4Xu2YS2gTURSG+/ItimisJG0mTaNIQUG6UNBN8YkKIj6gLhQXCuJKBHWhuKgUK1gRq6CIuHAj1FVBrMUHYheKSFEEdaGgVipaF335aG39T3Juufymk5k0agvzwWEm/7nn3HPPTCZ3kpcXEBAQ8P+IxWKTHcfZwnrAXwQN74cNse4HxP+SHBnsNMf5BTfIRc3Vg/M62D6cfywpKYnguAHWwDFeSFMr2+V4PL6A47LGJGbdLxUVFRNNruLi4mmQikRDQxLQbopeWVk5geO8gvgdkgONrsbHQqOXlZUVQ38J+xmNRrdZIX4oMrUnEokQbJLUjtyLMd8B9XVyUFagyHdmMvZlg+bqZV1QXyPrXkDcoFuNWMctN78X3PqAxi+VGnA8yz7fIFG722R+wMJnaa4W9gnqO8V6JpB3o8beZZ8BY86Mdg06Rz/rBview/pY9wUSNOnxk0wYCoWm8xg/IEe95MHdsCyNryGbpjj6WEJTD7PPBmMOwgZZ94o8lnSeWvYZ4L+WzRqGQfB62ICev9AJ4zzOD8jxFNaN0yLST0h+WI2te0HjhiKRyGz22aD2rRhXz7pXEHtBe7CafQb4W2QM655BcAd+jWfq+Q1Jhjt0E4/zisRqg2Qn1Afr0aPsamqyuaDyY6Y5s1+oR3Seh6wb0KvoqGpBg67jkG99PqkJj1nDfIEcz7Sg4V2FwdGvJcYcYZ8buFDbta4v7Ms1Mg/mW8m6Af7GbNaQBIFVupA7SHAbEzXj/I1q53i8VxyX3QX0zZr/M/vcQG2HNG7EH1ADxlSx5geZB1vbKawb4O+VMeXl5XPZlxEEfg2Hw3NIS14IXIQHtu4HbU7apiNvtfrfss8NNH251vWafTZycVjzA+J3jlS7AF+Z1vGYfRkpLS0NI3g/67jCi7Qpr9jnFY3vYV2A3qT+8+zLQL7GjbiNE+DvYs0PaPpVmYd1pRC+VlgnXpQcdmbEJfHw2xi+BVMt3Sx6pLgkuAN2y5g027oC6Ec1R7q7PJkf8evYYUDORxq/hH3y6i+/JXnW75ONx9oXag1/fMvR5DW6rmb2ZcRJPY+6nNRuos/efuHzE1i3+mXH8d2x3hpx/gHWbj7bQG9zUruV5POcbAB2DwUf5zgbzX+fdRvsYmZgzDfJK48aJ/Ve0R9L8z5gIxcE497L6zz7oF9yUms19YrJTkvq/gHrQO2rOO6fgcn3sJZLsMBW1hh5NEZTLzBXcNyLhs7jMenAt2F+uqaPaeQPKrlbWM8Vkh9NXMF6roiltsfjCzSkFk1vYz1XSH4cCljPBWj4Wif1hjy+QOG7WMslfzM/ctexFhAQEBAQEDDG+Q36GFy3biyd9wAAAABJRU5ErkJggg==>