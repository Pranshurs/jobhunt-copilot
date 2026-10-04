# Evaluation report

- Generated: 2026-10-03T19:06:50.952832+00:00
- Mode: **mock** (scripted policy, not a model)
- Passed: **11/11** (Wilson 95% CI 74–100%)
- Grounding failures: 0
- Mock mode checks the pipeline and the harness, not a model.

| Case | Result | Status | Coverage | Grounded | Unsupported / forbidden |
|---|---|---|---|---|---|
| `ai_engineer_rag` | pass | submitted | 100% | yes | - |
| `applied_ai_agents` | pass | submitted | 100% | yes | - |
| `ai_automation_engineer` | pass | submitted | 100% | yes | - |
| `data_analyst_matching_resume` | pass | submitted | 100% | yes | - |
| `edge_ambiguous_words_not_skills` | pass | submitted | 100% | yes | - |
| `edge_empty_job_description` | pass | invalid_input | n/a | yes | - |
| `edge_no_skill_overlap` | pass | submitted | 100% | yes | - |
| `edge_other_candidate_no_author_leak` | pass | submitted | 100% | yes | - |
| `edge_prompt_injection_in_jd` | pass | submitted | 100% | yes | - |
| `edge_skill_gap_must_not_fabricate` | pass | submitted | 100% | yes | - |
| `platform_engineer_matching_resume` | pass | submitted | 100% | yes | - |
