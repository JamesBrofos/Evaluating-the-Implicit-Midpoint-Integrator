# Stochastic Volatility Model

Examine the performance of the generalized leapfrog integrator to the implicit midpoint integrator on the hierarchical stochastic volatility model. To execute the job, run:
```
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 5 -t 24:00:00 --job-name stochastic-volatility -o output/stochastic-volatility-%A-%J.log --suppress-stats-file --submit
```
