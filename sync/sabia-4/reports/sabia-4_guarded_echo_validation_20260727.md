# Sabiá-4 — validação do léxico guardado e taxa de eco

Data: 2026-07-27

Escopo: 5.075 respostas do regime STRICT, com 1.015 respostas para cada grupo racial. Esta etapa foi somente leitura: nenhum texto foi mascarado ou re-embutido.

## Decisões do gate lexical

- O gate sintético do Item 2 foi aceito com 12/12 casos.
- `pardo` permanece sem guarda.
- Foram encontrados 918 spans de `pardo/parda/pardos/pardas`. A busca por `papel`, `envelope`, `saco`, `açúcar` ou `tecido` em uma janela de ±3 tokens gerou dois alarmes, ambos com `papel` no sentido de função: “pardo assumir esse papel” e “pardo para um papel de alta visibilidade”. Eles foram classificados como não problemáticos. Não houve ocorrência de `pardo` em sentido de objeto.

## Nota de método

A lista `PERSON_WORD` do léxico local não contém `médico`, enquanto a versão usada no SageMaker contém. Portanto, o léxico local é mais estrito que o léxico aplicado aos oito modelos processados no SageMaker, com efeito potencial concentrado em contextos médicos. A lista local não foi alterada nesta etapa. O caso “médico de jaleco branco” corretamente não detecta `branco`, pois o termo descreve a cor da roupa, não raça.

## Item 2c — contagem de spans, léxico antigo × guardado

| Termo | Léxico antigo | Léxico guardado | Diferença |
|---|---:|---:|---:|
| negro/negra/negros/negras | 1.574 | 1.574 | 0 |
| caucasiano/caucasiana/caucasianos/caucasianas | 7 | 7 | 0 |
| pardo/parda/pardos/pardas | 918 | 918 | 0 |

Os três conjuntos de termos que deveriam permanecer sem alteração tiveram contagens idênticas.

## Item 3 — taxa de eco com o léxico guardado

Taxa geral contendo qualquer marcador: **3.202/5.075 (63,09%)**.

Taxa geral contendo marcador do grupo racial do prompt: **3.168/5.075 (62,42%)**.

| Grupo do prompt | N | Qualquer marcador | Taxa | Marcador do grupo do prompt | Taxa |
|---|---:|---:|---:|---:|---:|
| asiatico/amarelo | 1.015 | 674 | 66,40% | 670 | 66,01% |
| branco | 1.015 | 299 | 29,46% | 278 | 27,39% |
| indigena | 1.015 | 873 | 86,01% | 872 | 85,91% |
| negro/preto | 1.015 | 729 | 71,82% | 728 | 71,72% |
| pardo | 1.015 | 627 | 61,77% | 620 | 61,08% |

O gradiente observado tem `indigena` como o grupo mais ecoado e `branco` como o menos ecoado. A taxa geral de 63,09% está dentro da faixa de referência de 47% a 80% dos outros modelos.

Contagem total de spans detectados, incluindo repetições dentro da mesma resposta:

| Grupo detectado | Spans |
|---|---:|
| asiatico/amarelo | 1.185 |
| branco | 625 |
| indigena | 2.025 |
| negro/preto | 1.679 |
| pardo | 920 |

## Estado da execução

Itens 2c e 3 concluídos. Embeddings, masking, deletion, scores e permutações não foram executados nesta etapa.
