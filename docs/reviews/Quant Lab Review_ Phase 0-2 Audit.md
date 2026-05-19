# **Institutional Audit Report: Quant Lab Phases 0, 1, and 2**

**Prepared for:** Claude Code (Development Agent)

**Author:** Lead Quant & Product Manager (Lopez de Prado Framework)

**Status:** Code Audit & Mandatory Refactoring Directives

## **Executive Summary**

Your software engineering scaffolding for Phases 0 through 2 is highly commendable. The use of uv, strict type hinting, and the modular agent architecture provides an excellent substrate.

However, a strict mathematical audit of the codebase reveals **four critical causal vulnerabilities** and **two structural flaws** that violate the *Advances in Financial Machine Learning (AFML)* framework. Financial ML requires absolute chronological and statistical purity; standard pandas or scikit-learn conventions often leak future data or destroy stationarity.

Do not proceed to Phase 3\. You are directed to immediately refactor the codebase to patch the following vulnerabilities.

## **1\. Critical Mathematical Vulnerabilities (Data Leakage & Stationarity)**

### **1.1. Vulnerability 1: The Triple-Barrier Intra-Bar Path-Dependency Violation**

* **Location:** src/afml/labeling/triple\_barrier.py  
* **The Flaw:** Standard retail implementations of the Triple-Barrier Method evaluate take-profit (TP) and stop-loss (SL) barriers strictly using the close price of the bars. If your implementation does this, it violates the core principle of path-dependency. A price could spike through the Stop-Loss barrier mid-bar, only to recover and close inside the barriers. Evaluating only the close results in massive look-ahead and survival bias.  
* **The AFML Refactoring Directive:** 1\. The Triple-Barrier evaluation *must* utilize the high and low prices of each bar to detect intra-bar barrier touches.  
  2\. Implement a deterministic **Conflict Resolution Rule**. If an extreme volatility bar causes both the high and the low to touch the Upper and Lower barriers simultaneously within the same index:  
  * **Mandate:** Treat any dual-touch bar as an automatic Stop-Loss hit (![][image1]). We must penalize ambiguity to enforce conservative risk modeling.

### **1.2. Vulnerability 2: FFD Constant History Length Violation**

* **Location:** src/afml/data/ffd.py  
* **The Flaw:** Fixed-Width Fractional Differencing (FFD) requires a constant history length (![][image2]) for all differentiated points. If the algorithm starts differencing from index ![][image3] by using an expanding window or filling the initial values with zeros/NaNs, the early rows of the dataset will have different statistical properties than the later rows (because they use fewer lag weights). This violates stationarity and corrupts the ADF test.  
* **The AFML Refactoring Directive:**  
  1. Determine the window threshold ![][image2] based on weight tolerance ![][image4].  
  2. **Strictly drop** the first ![][image2] rows of the output series. The resulting FFD dataset must consist *only* of points that have a complete history of length ![][image2].  
  3. Add a unit test asserting len(ffd\_output) \== len(input\_series) \- l\_star.

### **1.3. Vulnerability 3: Information Bar Update Leakage**

* **Location:** src/afml/data/bars/tick\_imbalance.py & src/afml/data/bars/tick\_run.py  
* **The Flaw:** In imbalance and run bar generation, the expected bar size ![][image5] and the probability of a tick being a buy ![][image6] must be dynamically updated. If you update these expectations using the *current* forming bar's ticks or within a non-causal rolling window, you inject look-ahead bias into the sampling frequency.  
* **The AFML Refactoring Directive:** 1\. The parameters ![][image5] and ![][image6] must be updated using an Exponentially Weighted Moving Average (EWMA) of the properties of **previously completed bars only**.  
  2\. The active, forming bar must use the EWMA values calculated at the exact close of the *prior* bar.

### **1.4. Vulnerability 4: Non-Causal Volatility Estimation**

* **Location:** src/afml/labeling/volatility.py  
* **The Flaw:** The standard deviation used to scale the Triple-Barrier limits must be calculated causally. If the volatility at index ![][image7] includes the price or return of the bar at index ![][image7], you are using future information to set the barriers for a trade entering at ![][image7].  
* **The AFML Refactoring Directive:** 1\. Calculate the daily rolling EWMA volatility of returns.  
  2\. **Strictly lag the volatility series by 1 period** (.shift(1)) before passing it to the CUSUM filter and the Triple-Barrier generator. The volatility metric used at time ![][image7] must be 100% determined by information available at ![][image8].

## **2\. Structural & Architectural Flaws**

### **2.1. The Alpha Registry Deduplication**

* **Location:** src/afml/core/registry/repository.py  
* **The Flaw:** To prevent duplicate tests from artificially inflating the Deflated Sharpe Ratio (DSR) in Phase 6, the Alpha Registry must mathematically prevent identical hypotheses from being logged twice.  
* **The AFML Refactoring Directive:** Ensure the SQLite schema has a UNIQUE constraint on the composite key (asset, algorithmic\_family, hyperparameter\_hash). The hyperparameter\_hash must be a SHA-256 string generated by sorting and serializing the hyperparameter dictionary (to ensure different key orders in JSON don't bypass the filter).

### **2.2. Event-Sampling Deduplication (Dual-Path Cleanliness)**

* **Location:** src/afml/core/events.py & src/afml/labeling/brain1.py  
* **The Flaw:** The dual-path mechanism allows events to be generated by CUSUM *and* native entry logic. If both trigger on the exact same tick, it will double-count the event.  
* **The AFML Refactoring Directive:** Implement a strict index deduplication layer. The final array of event timestamps (![][image7]) passed to Brain 2 must contain only unique, monotonically increasing datetime indices.

## **3\. Execution Orders**

1. Halt all new feature development.  
2. Refactor the codebase to address Vulnerabilities 1.1 through 2.2.  
3. Update the tests/unit/ suite to explicitly test for these causal and stationary fixes.  
4. Run the full test suite and confirm 100% passing status.  
5. Report back when the mathematical integrity of the pipeline is restored.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAACJ0lEQVR4Xu2WPUscURSGRyPGJo2NKLs7u5uVDfsLxJjUglVIE6wSDQhik0aFhJQiFhbW2lkF8gcUoigBIQFBUAiEQLQwQfGDgJ2uz3Hv6PEwo2uRYVfmhZe95z0f9xzmzp31vAQJEiS4QqPv+y/hWiaTmUin0x02oC6RzWY/MdQZfAF/wGMbU3fg6TxjkN+FQuFhoGEPwi2WjSq0vsAA83BOa7lcrg2tzJPs0noYHlihFkDjvW6Ad9YnOvxq9QsweRHnd/jH+tAWrRY36GHEDTYU4itzkWxb/QLyUsovQQcEjWmfJGq7GlBvg7yjO3DJ1tDAPyV9UPdNiK8stLrHIKVg7QKalN1vkhqw+5QdC+jxoxtgwPoiB1NosgHYs3BH2XLN3ljkf4A9h90Ab0N8oh9a/RI4Z+BeYPPe+VHFbgM57RybJ9Xytg8t9bqlF2LHQ3zS47LVL4FzHX4ObIq8liQ2fazjqgF5v9yG1XLT1jCQ07QPp7VIjy2Sz1Ed1fo1+JXBZpV9IEmyTqVSnay/UWiF39WrrPhA8z3s/Q+2K23Cj7rqNQj6C3fhoQwVDCZffa9ycchxeGXSYgN7T9LDqV+5JU/gTxsThQberefFYvGRG2xdO2Vgefxaixucnlb6mOFpPS2VSs3Wfw366Tj7CwMsmJgPaF1SWOs1DTfYiVsPiC3vlY7Juo+4nGut1zQY5D3cgbtyG3oR/5bz+XzGagkSJEhwr3AO5VCyzv7XJ7AAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAAzklEQVR4XmNgGKFASkpKRF5e/j8UX0OXl5OT60cXQwFABdpQzVEwMSB7LhA3A3ERUH6FgoJCIbIeOAAqyAFplpWVlUITXwrEn/DaDlTwGqQZTWwfUFMskM4C2toIxB3I8nAAdTK6Zk8QDTSgDso3RJaHA6jmq+jiIAAKUHQxOAA6xwCqOQhdjiAAhSJQ4z9paWlhdDmCAKjxPd7QxAdATgbaLgBlYw8UXAAWysbGxqzAePZDl8cLQJq1tLTYgPQJdDmiANDZ9uhio2AUgAAApkAw0nj61PYAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAZCAYAAABzVH1EAAAB7ElEQVR4Xu2VTSsFURjHr24hJVlI7tvce7u5uikWlqJ8AEpZWNlQIgs2FspCYseaHUnZyCfwEVhQYkeRhVhIXvL2e3JujacxzrgSml/9mznP25nnnDMzkUhISMifIJVKzTuO81KU28d4Bu26bb5QbJGEPW3/SaSJdDp9JPc8zxDjWzSK1tCDjvdEChC8qu0lEkVl2vgBZWZHlouGTCZTz/hSmkgmk23u4A8xq5HW9q/Aamapd4N6I5aNkDNmGmkw42nun3mkAa7r0pDO8USKaNtXYOINaj2iLu3zg/hNdFAcswPN7EgHtgm0HY/HE+74dxDYQudzTL4ijci9SMd9Bjl95N9RZ0n7bJDnMPP3aR+22mw226jtnphGDrXdj3w+X03OGTpnohrtDwIPu/UtJ4JCJ0FWk0ln0RU5w6hS+4PivL3QpTVCgUEp4piXzAZp2hylnojly+yHmf9J2wNBgVP3arA77W6/H3J+yZ1Cx7FYrEr7LYmaRhakHprUAVaYIsVGpOj+uwALyBlBF/LJ1L7PkP+DzM+1mx0e535Hx1hB4rUUkrPO9TSRSOR0TBCosY3uqdekfV7IF8k0Ig1d5HK5Oh1jTaFQKGfiVm0vBfNTvKZuv/Z5ECWuUxt/DfKJZoUrtD0kJCQk5F/wCpm5ekSEpjM+AAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE8AAAAaCAYAAAD2dwHCAAACgElEQVR4Xu2WPWsUURSGExM1rWCT/Zpd3EK2EVlQCFpEUWwUvxEJxiQIflSxUzAgFkEbEbQV0cYf4C8QRVEIikhMkyKFkGAhBgyJaHwOe0cOh73jxLC7jNwHXmbue879OntnZru6AoH1Ui6XhyuVykG5z+fzhWKxeMTmBDxEUTSDVtE3uVK8vTYn4IGCTcuJs37mYWPP0D7rOzYQO4FelUqlyUKhkLcJaaD/x3KDm4wzZONZo1tOApt67B6nszZBYKNPXfwY+iSPnc1JA/0+oCn5ISjgc6mizckMbGKUF/guinPSVzxie9Ac++yLPe7PkzvDba9rj5Mz4RO5O/8MqJA5rZc5koqH9wQ91F4ul9sq+VJYaXM9QPu4T/Fjzv1gvV7fGI/zXxePE3XI+Ve1L4hP/I31k6DPvBRatX/peCI8IuRH79B01Hh3xJIX6W6b3y58xaN9xRXpovZdTPI/Wz8J8l/Gj7+cWtoPbE5TSPyB3tLpFHrE/RLX07JwBjxq8w3d5L9HX9PKDpCEr3is67bzR7UvOH/Nj121Wt3MuGf4j7fNxprCiduh20y6hO5rr5PExSuZvw94N1yRxrTvYv9UvHXj+zU7ha94nJBLzr+gfcHtYdH6raaXSZdrtdomG0iCPv1sZnta2f5JqOKd0z7egPiMd037LibFe2H9lsIC9zPplPX/grzzZt2CU8kOkERcPIo0bEI9+F+I39Wm/PBunuvabzksZI5JL1u/k6jijTSJyVdxEfXHHve30Gud1xZkkfLFsX4nYC0rUePrvIDm3VXaKzqPAk7i/UR30Hc0q+Ntg4Uctl4WYN1bOJn35CSu9X0dCAQCgUAgEAhYfgOl3NzhD/NalQAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAZCAYAAABzVH1EAAACh0lEQVR4Xu2WTahNURTH75OPyES6k/t17u3e0rs+Jrf0rokBhZIwIULoDZSPMjA1U+qNZGJigImBMhGFgYFkghCGMlKUkTd4PPxWd213tTrvnn2ePO/l/mt19lr/tdfa/332+SgUhvjPkCTJQbVRz8XC1NjhuTkDzT9jb7BjnouFzhd75rlUNBqNNST/jLFisbjSz09DrVa7YP16vb7N18qwTWFuEiskgAnXpAiLGPecQDgfmwleCHO/lsvliotNYZOtVmuZi39yfj4hNP8gi61Wq03PCYT3sZmQIuSb9TUmO//Ax5n7yPq5hJB8XAtftXGOxEWT88Jyg2CFlEqlFfhbLC+QftQfS4mfdH4uITekMA0PhRhHYTWxKZNzP4yzYIWw2M1cRgxdkGdN+jFcbOMC4l3n5xIid0Osy8O/kete7Dt2x+fGwB8tD+pOqJBMzFbIaKVSWc91j/gsqONzYzBICNwqak9jXzyXhjxCRlTEDxvURouCL0ctjDku+1nQ+Xa7vTTELAYJYe5u3aTbnktDtBAKnlYhl1x8axjzJluHv7bT6Swh72OIM34cxhaDhDDnlfTjCG/wXBqihcjOy92wO24B18WmdXxLFhE4dncf/tl+dg8ZQmTTop4PQR4h8hp86eMB8O9q+g1h/MQugvhO5l7pZ/+Oz50Qmo1j51TITfG5HsGO6nF7rk0nw91Kev9Q9o5sJ/dev2oPXojUJXYq0b8H7XmC6wH/ZffIFDIbsJinVgjPzi78yzZH4IX8Cf6KEIpet0JY8GHsjM3R+PwWwjdmOYVfB5/xW8sHzHshgmazWaX4XbkT8mx5XrAghMSA5u+xh/J69lwsZL7avxMyxBALHL8AKQrZhzXUaZQAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAaCAYAAAAg0tunAAACn0lEQVR4Xu2YzatMYRzHr3u9pCipqVtm5plhNBps7ihLFkpZ2ZEdITssLfwBysZG7JQFq3G7pBA2Fogoke5CVuSKqOuluF2fZ+Y5enw755jJmZnn5nzq1znP7+15nu859znTHRnJycn53zHGtLAX2AGNLVQqlcpOt6c3GkuEonEK5lNsDrtG6iK/Dl/LH9PniFdz0o+FSLlcPqS+CGKH1ZdKo9FYStFRt/mWHddqtWX1en0l4+vWj0Dn/RoV0LHY5W7TQCgUi8XNrPG5XafGInoW0ELDJ7YpE6zRGIKc0QkTBLT+xIUNG/axu1qtmmivGo/oWUAar6LhHNdpjVloeFknjBOwUCis0LwQyVxAmp22DRFwr8ZKpdImG8N++v44Aak/hX+W21He5PUaD4V+CPjYCTiuMZpdcLHjvj9BwIf4b5nOl+w745s8gHWaN2z6IaB9w+bZ7BauE2x8K3aC+0fYa+63x9T8ISDjXa7Pfc/XfjB+Xje4c+pTL6Y90jD9EtB+ea3hGtUcxYiATPpAF4XvhvpCIFMBEazgBHyrsTRUQMY/dFGM36sv7pwdNJkKSPI5J+BBjaURI6Dt8SXG93uhiHeM+Xb4OUmQu6EX0/o0MhWQyadtM4rWaiyNBAGfxvhmvPEVvs7L/Zw43BnYFr9b0x5pmCwFtI3sl1b9f8PECxi9gWPc3+PhTNmB+324z83VGPaXORMBafAVm8U+Y99M5wyb0LwkVMBms7kE30XsFfYM8fb7cXxX7Ty+b9Aw/1nsI/bO2Qz2Abvk53Ul4L+iAkYknXGm83PjjvpDZKgCJkF++58MnIE1Fjip8ZAIVkDEW831LgvcqPGQCFJAGLNvnzpDZGACYrf5s9yjsYWKPWLsnrCXGsvJycnJ6fALzccNa+Ifgn0AAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAcCAYAAAC6YTVCAAABAElEQVR4XmNgGAWYQF5eXhOIp6CL4wVADUeA+D+6OF4A1PCHHE3/FRQUbqGL4wUgTXJyctPRxTEAUKERUGE70IZNIE1A3A3kt6GrwwqAipuB+B66OE4AtIUDqOEnECehy4GAlJSUCJBiQhEEarIHOU1GRkYFRQIKgHKbZWVlldEF95AT1F+RNQFtzkSWxwqgofYdxNbS0mIDsmtAbGAopgGxElZXAAV/AfEfFRUVUaCiFcbGxqxQ8fNAW8OxagIBUVFRHgzPMoA1/gbiyejieAHIFkVFRTNgsHOhy+ECjEDndQKdzQdio0viBECbrgD9eRBdHC8AOk0e5F908VEABAA1tzilF1Fh0wAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAaCAYAAADWm14/AAABTElEQVR4Xu2Tu0oDQRSGgwS1sYiVsPdlcXEbQRSxEgsfwEshSKo8hYUWvoCNjVhY+hQ+gE+gYmElYm0hWGT9DszC5JDIhJhFZD/4mZ3/XOZkdtNqNTT8JcIw7ERRdKD92uDwF3St/drg8BIda782ZADtTZ0gCNY5+NFIbuAZPem8qcOhPdTXfm1w+Cu61L6AvxbH8bb2XeCf1aV2SfsDpGm6LNdP8qGOCcS+0IP2f8L3/UX63UhfXvOGjg9A0q0kav83cB3gzR6A5x07PgmuA5TVAFzbBbo3oTbPBbHPPM8XrBJnnAaQD4XED/TOu8sqn/2K53k+H9FV5Zl3ezZM5J+ydqpc06OkftP2hpJl2VySJKvap/iEJlvWfp79/gjtFUUxa9c7DzAKaWDWOx1zQertHzA20oBr3WU91zEXzA0caX8c2mhGmw0N/4ZvHKpQ4JIqCyoAAAAASUVORK5CYII=>