# Fitzhugh-Nagumo ODE Model

Examine the performance of the generalized leapfrog integrator to the implicit midpoint integrator on the Fitzhugh-Nagumo differential equation model. To execute the job, run:
```
dSQ -C cascadelake --jobfile joblist.txt -p week --max-jobs 1000 -c 1 -t 7-00:00:00 --job-name fitzhugh-nagumo -o output/fitzhugh-nagumo-%A-%J.log --suppress-stats-file --submit
```
