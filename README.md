![CrowdStrike FalconPy](https://raw.githubusercontent.com/CrowdStrike/falconpy/main/docs/asset/cs-logo.png#gh-light-mode-only)
![CrowdStrike FalconPy](https://raw.githubusercontent.com/CrowdStrike/falconpy/main/docs/asset/cs-logo-red.png#gh-dark-mode-only)


# _PyMalEvasion_: Generative AI-based Adversarial Evasion in Python Scripts

Data is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15430080.svg)](https://doi.org/10.5281/zenodo.15430080)

## Dataset
| Name | Type | Clean | Dirty | Total |
| ---             | ---    | ---    | ---       | ---    |
| Ours (train)    | Script | 15,242 | 11,456    | 26,698 |
| Ours (valid)    | Script | 1,075  | 2,046     | 3,121  |
| Ours (test)     | Script | 603    | **1,760** | 2,363  |
| Ours (all)      | Script | 16,920 | 15,262    | 32,182 |
| Ours (adv_test) | Script | 0      | **5,332** | 5,332  |

The _PyMalEvasion_ dataset is constructed by augmenting the [PyPI Malregistry dataset](https://github.com/lxyeternal/pypi_malregistry) with samples from VT. You can use [this script](./dataset/pkg_extract.py) to extract the sources from the archived PyPI Malregistry. After extracting the sources, we filtered out those under 512 bytes (e.g. containing typically harmless initialization or configuration scripts).

We further split the data into train/valid/test following a [cluster-informed method](./dataset/cluster_split.py). We apply the shallow FX, UMAP for dimensionality reduction and HDBSCAN for the actual clustering. Finally, the splits are chosen such that all samples in a cluster are from a single split, thus minimizing potential information leakage.

## Adversarial generation
- Heuristics (simple modifications to add comments, documentation, padding)
- LLM constrained via [AST](./llm_based/generate_constrained.sh) and [RAG](./llm_based/rag_generation.sh)
- [LLM unconstrained](./llm_based/generate_unconstrained.sh)

For [AST-based constrained generation](./ast_impl/ast_impl.py), the LLM is instructed to generate an action (add/edit/delete) and a code snippet for which the action to take place. Then, the AST of the original script is updated from the (smaller) AST of the snippet.

## Classifiers

We employ 3 classification strategies: [shallow](./classifiers/shallow/) (XGBoost on handcrafted features), CodeBERT (adapted from [microsoft/CodeBERT](https://github.com/microsoft/CodeBERT), base model: [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)) and [LLM-based](./llm_based/llm_classifier.sh).

For the shallow classification we built [8 feature types](./classifiers/shallow/shallow_fx.py) and [trained](./classifiers/shallow/shallow_trainer_cv.py) an XGBoost model for each one of the 255 feature combinations. Models are trained with HPO and 5-fold cross-validation.

## Support statement

_PyMalEvasion_ is an open source project, not a CrowdStrike product. As such, it carries no formal support, expressed or implied.
