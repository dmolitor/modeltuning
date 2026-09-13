## R CMD check results

0 errors | 0 warnings | 0 notes

* This is a new release.

---
This release is resolving an issue where we imported the full
`paws` package and this was excessive, causing big delays in reverse
dependency checking. I have trimmed down to the minimal sufficient
`paws.compute` sub-package, so this ought to resolve the outstanding
issues.