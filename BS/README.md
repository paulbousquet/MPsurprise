It's unfortunately not possible to do an automatically updating series of Bauer and Swanson because some of the data cannot be pulled automatically. Instead, I provide some materials to facillitate the replication. 
* `bs_replication.py` provides the controls necessary to replicate the BS series
  * takes an input `dates.csv`, the dates you want to calculate the controls for
  * `skew.csv` is yield curve skewness ([Bauer and Chernov](https://www.frbsf.org/research-and-insights/data-and-indicators/treasury-yield-skewness/))
  * `base.csv` is the original BS data as a fallback.
