# FiCi

**FiCi** (*Fictitious Citations*) is a lightweight Python package for detecting fabricated or hallucinated citations in scientific PDFs. It's tuned for standard single-/double-column conference layouts (NeurIPS, ICLR, ACM `acmart` / SIG conf) and avoids LLMs or heavy ML models.

## Install

From PyPI:

```bash
pip install fici
```

From source (editable, for development):

```bash
git clone https://github.com/sadjadeb/fici.git
cd fici
pip install -e ".[dev]"
```

## Command-line usage

Installing the package registers a `fici` console script:

```bash
fici paper.pdf --email you@example.org
```

Useful flags:

```bash
fici paper.pdf --email you@example.org --workers 8          # more concurrency
fici paper.pdf --email you@example.org --json > out.json    # machine-readable stdout
fici paper.pdf --email you@example.org --save-output         # Markdown report in cwd
fici paper.pdf --email you@example.org --quiet              # summary only
fici --help
```

`--save-output` writes a Markdown report to the **current working directory** as `paper-fici-YYYYMMDD-HHMMSS.md` (using the PDF file’s basename and a local timestamp). It is written **in addition** to normal stdout, so you can combine it with `--quiet` or `--json` freely. Use `--json` when you need machine-readable output; `--save-output` is for the human-readable report file only.

The CLI returns a non-zero exit code if any citation is flagged, which makes it easy to drop into CI pipelines:

| Exit code | Meaning                                                |
|-----------|--------------------------------------------------------|
| `0`       | All references verified.                               |
| `1`       | At least one reference is flagged or errored.          |
| `2`       | Bad input (e.g. PDF not found).                        |

`python -m fici ...` is equivalent to the `fici` script if you haven't added your Python `bin` directory to `PATH`.

## Programmatic usage

```python
from fici import FiCiPipeline

pipeline = FiCiPipeline(email="you@example.org")  # polite pool
reports = pipeline.run("paper.pdf")

for r in reports:
    print(r.index, r.verdict.value, round(r.score, 1), r.suspected_title)

print(FiCiPipeline.summarize(reports))
```

See [`example.py`](./example.py) for a complete programmatic usage example.


## How it works

The pipeline has four phases, each exposed as a standalone class:

1. **Extraction** (`ReferenceExtractor`): PyMuPDF pulls text, heuristics locate the *References* / *Bibliography* section, and regex splitters handle the dominant reference styles (`[1] ...`, `1. ...`, Author-Year).
2. **Structuring + Search (primary)** (`CitationSearcher.search_openalex`): each raw citation is sent to the [OpenAlex](https://docs.openalex.org/) `/works` endpoint as a free-text query (title only, for precision), using the polite pool via `mailto`. The hits are then handed to the verifier.
3. **Search (second opinion)** (`CitationSearcher.search_crossref`): whenever the OpenAlex-based verdict is anything other than `Verified` (suspicious match, no match, or error), FiCi also queries Crossref's `query.bibliographic` endpoint and verifies its hits.
4. **Search (preprint fallback)** (`CitationSearcher.search_arxiv`): if Crossref also fails to verify, FiCi queries the [arXiv API](https://info.arxiv.org/help/api/index.html) with a title-scoped phrase query (`ti:"<title>"`). This catches preprints that neither OpenAlex nor Crossref have fully indexed. The pipeline then returns whichever of the (up to) three reports is strongest — `Verified` always beats other verdicts, and within the same tier the higher score wins. If any earlier backend verifies, the subsequent ones are skipped to save latency.
5. **Verification** (`CitationVerifier`): `rapidfuzz.fuzz.token_sort_ratio` compares the API-returned title to the suspected title in the raw string, with a small bonus for corroborating author surnames. The pipeline emits one of three verdicts:

   | Verdict               | Condition                                                                                                       |
   |-----------------------|-----------------------------------------------------------------------------------------------------------------|
   | `Verified`            | Score ≥ verify threshold (default **90**).                                                                      |
   | `Suspicious/Mismatch` | Score below the verify threshold **or** none of OpenAlex / Crossref / arXiv returned any hits at all. Inspect `report.reason` to distinguish a low-score match from a no-hit "likely hallucinated" reference. |
   | `Error`               | API call raised an unrecoverable exception.                                                                     |

## Tuning knobs

- `FiCiPipeline(verify_threshold=90)`: single cutoff — scores at or above it are marked `Verified`, everything else `Suspicious/Mismatch`. Raise it for stricter verification, lower it for higher recall.
- `FiCiPipeline(max_workers=4)`: API calls are dispatched concurrently via a thread pool (I/O-bound work). Default is **4**, which stays under the OpenAlex / Crossref polite-pool rate limits. Set to `1` to force sequential execution, or override per-call with `pipeline.run(pdf, max_workers=N)`.
- `CitationSearcher(max_results=5, timeout=15, retries=2)`: control API politeness and robustness.
- Inject a custom `ReferenceExtractor` subclass if you need to support a non-standard template (e.g. workshop-specific layouts).

## Current limitations

- Title extraction from raw strings is heuristic; unusual punctuation or missing years can occasionally yield an incomplete `suspected_title`, which is why scoring also consults the full raw string.
- Author matching uses surname containment rather than a structured parse. If you'd like structured parsing via `anystyle` or GROBID, that's a clean extension point on `CitationSearcher._prepare_query`.


## Todo

- [ ] Add batch mode to the CLI to process multiple PDFs at once.
- [x] Add option to save the Markdown report to a file (`--save-output`).
- [ ] Add support for other templates (e.g. COLM, ACL, etc.).
- [ ] Add method to use doi directly to check if the paper is real.
- [ ] Add support for direct url to the paper as input.
- [ ] Add support for .bib file as input.
- [ ] Add support for single citation as input.
- [ ] Set up the package on HF space.
- [ ] Add Github Actions to the package.
- [ ] Add Github page for the package.
