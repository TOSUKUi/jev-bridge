# Task for researcher

Answer with primary sources (LiteLLM docs at docs.litellm.ai, BerriAI/litellm GitHub code/issues — cite URLs).

Context: a client POSTs to an OpenAI-compatible {base_url}/chat/completions with this body:
{"model": "...", "messages": [...], "temperature": 0.0, "max_tokens": 1, "logprobs": true, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": false, "preserve_thinking": false}}
and reads choices[0].logprobs.content[0].top_logprobs = [{token, logprob}, ...]. The top-K logprob list must survive a LiteLLM proxy round trip (BerriAI/litellm, current version ~1.10x).

Answer these, each with a citation:
1. Does the LiteLLM proxy return logprobs / top_logprobs for chat completions? Which providers support it (look for supports_logprobs / supports_top_logprobs capability flags in the supported-params code and the model cost map)?
2. If a provider does not support logprobs: 400 error, silent drop, or error unless drop_params=True? What is the default of drop_params?
3. Are unknown/non-standard top-level body params such as chat_template_kwargs forwarded to an upstream OpenAI-compatible server when the deployment is declared as model: openai/<name> with an api_base? If not forwarded by default, what documented setting makes it pass through (extra_body, openai_extra_params, additional params passthrough, wildcard config)?
4. Known bugs: search GitHub issues for litellm + logprobs ("logprobs not returned", "top_logprobs dropped", "logprobs proxy"). Open or fixed? Which release?
5. Anything about logprobs being stripped when LiteLLM transforms a response (non-OpenAI providers, guardrails, logging hooks).

Output format, keep it short:
VERDICT: works | works-with-config | broken
CONFIG: minimal litellm config.yaml snippet if any setting is needed
CAVEATS: bullet list, each with its source URL

---
**Output:**
Write your findings to exactly this path: /home/amemiya/work/jev-bridge/.pi-subagents/artifacts/outputs/b3771df8-1079-49d6-b4dd-27292c42a1b0/research.md
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```