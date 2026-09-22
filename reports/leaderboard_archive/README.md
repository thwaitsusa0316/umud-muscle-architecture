# Leaderboard snapshot archive
Historical `reports/leaderboard_<UTC>.csv` snapshots older than 24 h are moved here as `<UTC>.csv`
(prefix stripped) on 2026-09-22T03:45Z so that `masters/tools/pre_submit_challenge.py`, which
treats every `leaderboard*.csv` under the workspace as a live read, measures only the fresh
snapshot. A PLAN.md / ledger reference `reports/leaderboard_X.csv` resolves to
`reports/leaderboard_archive/X.csv`. Nothing was deleted.
