# R-BIAS data

`scenarios/rbias_scenarios.json` is the canonical grouped source used by the evaluation pipeline. It contains 263 scenarios and five prompt variants per scenario. `prompts/rbias_prompts.csv` is a derived flat view; it must be regenerated from the canonical JSON rather than from older working CSV files.

The source dataset retains two historical labels used by the experiment cache:

| Source label | IBGE category represented in public records |
|---|---|
| `negro/preto` | `preto` |
| `branco` | `branco` |
| `pardo` | `pardo` |
| `indigena` | `indígena` |
| `asiatico/amarelo` | `amarelo` |

The composite source labels are not additional racial categories. They are preserved so scenario identifiers, cached filenames, and prior analyses remain traceable.

The strict counterfactual regime contains 203 scenarios from six categories. Its five versions are validated after replacing known gender and number forms of the racial marker; all other wording and punctuation must remain identical. The implicit signaling regime contains 60 scenarios from `historico_e_merito` and `estereotipo_implicito`. Its culturally racialized signals concern perceived coding in Brazilian usage. They do not determine the race of any real person.

`schemas/` contains machine-readable schemas for source scenarios and canonical response records. `samples/` contains invented scenarios and one response per group under `samples/responses/` for tests and dry runs; it contains no production responses or findings.

Raw model responses, embeddings, provider caches, annotation records, and third-party datasets are not distributed from this directory. Their redistribution and provider terms require separate review. Original benchmark data and schemas in this directory are intended to be covered by `DATA_LICENSE`; author review is required before public release.
