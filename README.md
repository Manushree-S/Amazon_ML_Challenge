# Amazon ML Challenge 2026: Business Entity Resolution

Complete, runnable repository for the Amazon ML Challenge 2026 Business Entity Resolution task.

## Repository Overview

```
.
├── output/
│   ├── matching_results.tsv        # Scored on leaderboard
│   └── candidate_pairs.tsv         # Blocking's final candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Python pipeline modules
│       ├── README.md               # End-to-end reproduction guide
│       └── requirements.txt        # Pinned dependencies
├── Documentation_template.md       # Methodology report
├── utils/
│   └── validate_submission.py      # Format validation script
└── .gitignore
```

For detailed execution and reproduction instructions, refer to [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md).

## Validation Command
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
