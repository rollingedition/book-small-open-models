# BANKING77: data card

A copy of the BANKING77 intent dataset: customer messages to an online bank, each labelled with one of 77 intents.

## Source

- Repository: https://github.com/PolyAI-LDN/task-specific-datasets
- Commit: `57ec275d8078af65b7731c2a98be812d844a6d6b` (committed 2022-04-29)
- Path in that commit: `banking_data/`
- Copied without change: `train.csv`, `test.csv` and `categories.json` from `banking_data/`; `LICENSE` from the repository root (`banking_data/` has no `LICENSE` at that commit)

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0). The `LICENSE` file here is copied byte-for-byte from the root of the source repository at the commit above. Share and adapt with attribution, as the licence sets out.

## Citation

Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., and Vulić, I. (2020). Efficient Intent Detection with Dual Sentence Encoders. In *Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020*. https://arxiv.org/abs/2003.04807

## Contents

| File | Rows | What it holds |
| --- | --- | --- |
| `train.csv` | 10,003 messages (plus a header row) | columns `text`, `category` |
| `test.csv` | 3,080 messages (plus a header row) | columns `text`, `category` |
| `categories.json` | 77 labels | a JSON list of the intent names |

- Labels: 77 intents. Every label appears in both `train.csv` and `test.csv`.
- The split is the source's fixed train and test split. It is not re-drawn here.

## Files and SHA-256

```
7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661  LICENSE
53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63  categories.json
d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d  test.csv
b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b  train.csv
```

## Personal details check

Every message in both splits (13,083) was scanned before shipping.

| Kind | Count | Notes |
| --- | --- | --- |
| Email addresses | 0 | |
| Runs of 8 or more digits (card or account numbers) | 0 | |
| Phone-number-like strings | 0 | |
| IBAN-like strings | 0 | |
| Street addresses and UK postcode-like strings | 0 | |
| Web addresses and handles | 0 | |
| A child's first name | 1 message | in a transfer message, in `train.csv` |
| A pet's name | 2 occurrences | in customer messages |
| Town, city or region names | several | towns and countries, not addresses |

The name check looked at capitalised words inside sentences. It cannot find a name written in lower case or at the start of a sentence, so the scan is a check, not a guarantee.
