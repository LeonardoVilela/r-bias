# Notebook inventory

Notebooks are not the canonical implementation. Reusable metric, validation, generation, and reporting logic belongs in `src/rbias/`.

The local research workspace contains six historical notebooks: `abliteration.ipynb`, `dataset_generator.ipynb`, `sentence_bias.ipynb`, `sentence_bias_transformers_fixed.ipynb`, `weat.ipynb`, and `weat_transformers_fixed.ipynb`. Several contain stored outputs and credential-related variable names. Their originals remain unchanged outside this Git repository.

They were not copied into the release candidate because output clearing, credential review, third-party dataset licensing, and a mapping from notebook cells to published analyses require author confirmation. Before adding any notebook:

1. preserve the original outside Git;
2. remove credentials, private URLs, local absolute paths, and private outputs;
3. clear execution outputs unless they are small, public, and necessary;
4. state which result or diagnostic the notebook reproduces;
5. import reusable code from `rbias` rather than duplicating formulas;
6. verify the licenses of every embedded dataset or excerpt.

TODO: identify which first-part WEAT, SEAT, and SBS notebooks are intended for public archival and document their input licenses.
