# Sabiá-4 — masking definitivo com léxico guardado

Data: 2026-07-27

## Protocolo

- 203 tuplas STRICT, cinco grupos e cinco amostras por grupo.
- Variantes: `mask` e `deletion`.
- Encoders: `BAAI/bge-large-en-v1.5` e `intfloat/multilingual-e5-large-instruct`.
- Embeddings calculados na GPU NVIDIA GeForce RTX 4060 em fp32.
- 10.000 permutações por combinação, seed 42.
- Ratio reportado: `score / ((ci_low + ci_high) / 2)`.
- Bootstrap: 10.000 reamostragens das 203 tuplas, seed 42.

A função de null analítica `_analytic_null_floor(σ², G, N)` existe e é chamada pelo caminho de score. Por isso, os JSONs STRICT também contêm `null_floor_analytic` e um campo legado `signal_to_noise_ratio` calculado com esse null analítico. Esses campos não são o ratio apresentado abaixo. O ratio do protocolo usa exclusivamente o ponto médio do IC95 da permutação.

Para evitar uma comparação fp16 × fp32, as respostas originais sem masking foram re-embutidas em fp32 em um caminho novo. O score BGE original fp32 foi 0,021059, contra 0,021068 no resultado histórico.

## Item 4 — resultados definitivos

Todos os resultados contêm 203 tuplas. Em todos os quatro testes, nenhuma das 10.000 permutações atingiu o score observado (`p_value = 0` no código atual).

| Variante | Encoder | Score original fp32 | Score após controle | Queda | Null CI95 | Null floor | Ratio | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mask | BGE | 0,021059 | 0,018355 | 12,84% | [0,014419; 0,014975] | 0,014697 | 1,249 | 0/10.000 |
| mask | E5 | 0,007031 | 0,005419 | 22,93% | [0,003877; 0,004050] | 0,003964 | 1,367 | 0/10.000 |
| deletion | BGE | 0,021059 | 0,018071 | 14,19% | [0,014352; 0,014901] | 0,014626 | 1,236 | 0/10.000 |
| deletion | E5 | 0,007031 | 0,005337 | 24,09% | [0,003855; 0,004027] | 0,003941 | 1,354 | 0/10.000 |

### Correlação por tupla

| Variante | Encoder | N | Pearson r | Spearman ρ |
|---|---|---:|---:|---:|
| mask | BGE | 203 | 0,9385 | 0,9129 |
| mask | E5 | 203 | 0,9313 | 0,8378 |
| deletion | BGE | 203 | 0,9399 | 0,8940 |
| deletion | E5 | 203 | 0,9276 | 0,8337 |

### Maior par antes e depois

| Variante | Encoder | Maior par original | Distância | Maior par após controle | Distância |
|---|---|---|---:|---|---:|
| mask | BGE | branco – indigena | 0,244125 | branco – indigena | 0,219602 |
| mask | E5 | negro/preto – indigena | 0,138020 | negro/preto – branco | 0,120697 |
| deletion | BGE | branco – indigena | 0,244125 | negro/preto – branco | 0,217090 |
| deletion | E5 | negro/preto – indigena | 0,138020 | negro/preto – branco | 0,119643 |

## Item 5 — margens dos maiores pares, mask × BGE

Ordem dos grupos nas matrizes: `negro/preto`, `branco`, `pardo`, `indigena`, `asiatico/amarelo`.

### Original fp32

| | negro/preto | branco | pardo | indigena | asiatico/amarelo |
|---|---:|---:|---:|---:|---:|
| negro/preto | 0 | 0,225654 | 0,209516 | 0,236318 | 0,227590 |
| branco | 0,225654 | 0 | 0,195859 | 0,244125 | 0,206626 |
| pardo | 0,209516 | 0,195859 | 0 | 0,233977 | 0,204949 |
| indigena | 0,236318 | 0,244125 | 0,233977 | 0 | 0,239986 |
| asiatico/amarelo | 0,227590 | 0,206626 | 0,204949 | 0,239986 | 0 |

- Primeiro: branco – indigena, 0,244125.
- Segundo: indigena – asiatico/amarelo, 0,239986.
- Margem: **0,004139**, abaixo de 0,01.

### Após mask guardado

| | negro/preto | branco | pardo | indigena | asiatico/amarelo |
|---|---:|---:|---:|---:|---:|
| negro/preto | 0 | 0,219232 | 0,203416 | 0,214524 | 0,215629 |
| branco | 0,219232 | 0 | 0,195815 | 0,219602 | 0,198747 |
| pardo | 0,203416 | 0,195815 | 0 | 0,210288 | 0,196943 |
| indigena | 0,214524 | 0,219602 | 0,210288 | 0 | 0,213719 |
| asiatico/amarelo | 0,215629 | 0,198747 | 0,196943 | 0,213719 | 0 |

- Primeiro: branco – indigena, 0,219602.
- Segundo: negro/preto – branco, 0,219232.
- Margem: **0,000370**, abaixo de 0,01.

### Bootstrap dos dois maiores pares

| Estado | Par | Média | IC95 |
|---|---|---:|---:|
| original | branco – indigena | 0,244125 | [0,235155; 0,253692] |
| original | indigena – asiatico/amarelo | 0,239986 | [0,231978; 0,248676] |
| mask guardado | branco – indigena | 0,219602 | [0,212123; 0,227512] |
| mask guardado | negro/preto – branco | 0,219232 | [0,212070; 0,226764] |

As distâncias dos dez pares para cada uma das 203 tuplas foram salvas em `pairwise_distances_per_tuple_bge_mask.csv` (2.030 linhas).

## Item 6 — efeito isolado da correção do léxico

Esta comparação re-embute tanto o léxico antigo quanto o guardado em fp32. Assim, a diferença abaixo isola a mudança lexical. Os resultados históricos foram preservados separadamente e não são usados para estimar esse efeito.

| Variante | Encoder | Score antigo | Score guardado | Mudança | Ratio antigo | Ratio guardado | Mudança do ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| mask | BGE | 0,018299 | 0,018355 | +0,31% | 1,2465 | 1,2489 | +0,0024 |
| mask | E5 | 0,005405 | 0,005419 | +0,26% | 1,3647 | 1,3673 | +0,0025 |
| deletion | BGE | 0,018060 | 0,018071 | +0,06% | 1,2351 | 1,2355 | +0,0004 |
| deletion | E5 | 0,005335 | 0,005337 | +0,04% | 1,3538 | 1,3543 | +0,0005 |

## Item 7 — nota de truncamento

A distribuição de comprimento do Sabiá-4 — p10 1.553, mediana 1.767, p90 1.921 e máximo 2.263 caracteres — é estreita e comprimida contra um teto, comportamento compatível com truncamento no limite de 512 tokens, e não com variação natural de tamanho.

As respostas não contêm contagem de tokens gerados nos metadados. Portanto, não foi possível calcular a porcentagem que atingiu exatamente 512 tokens.

## Arquivos principais

- `results.json`: resultados completos e metadados.
- `item4_scores.csv`: scores, nulls, ratios e p-valores.
- `item4_tuple_correlations.csv`: Pearson e Spearman por combinação.
- `item4_largest_pairs.csv`: maiores pares antes e depois.
- `pairwise_distances_per_tuple_bge_mask.csv`: dez pares × 203 tuplas, antes e depois.
- `top_pair_bootstrap_ci95.csv`: IC95 bootstrap dos maiores pares.
- `item6_old_vs_guarded.csv`: comparação lexical controlada em fp32.
- `item6_historical_confounded.csv`: comparação histórica preservada, marcada como confundida pela precisão.
