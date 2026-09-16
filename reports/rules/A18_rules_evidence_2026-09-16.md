# A18 evidence — may the fascicle detector be trained / pseudo-labelled on the 309 provided test images?

Read 2026-09-16T01:31Z by headless Google Chrome (`--headless=new --dump-dom`) — the first full machine read of the Rules page (curl and WebFetch had returned only the client-rendered shell).

Sources (full text):
- /Users/smt2/data/competitions/umud-muscle/reports/rules/rules_text_2026-09-16T01-31Z.txt (Rules page, 35 KB)
- /Users/smt2/data/competitions/umud-muscle/reports/rules/overview_text_2026-09-16T01-35Z.txt (Overview page)
- Raw DOM: /Users/smt2/data/competitions/umud-muscle/reports/rules/rules_dom_2026-09-16T01-31Z.html

## Verdict: no clause forbids it. A18 verified.

Every clause in the Rules that touches data use, labels, or training, quoted:

1. **s2.4.a Data Access and Use** — "You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education." The test images are Competition Data (s3.18.a: "The Competition Data will contain private and public test sets").
2. **Foundational Rule 4.b** — "Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records." Automatic pseudo-labelling by the model is neither hand labeling nor human prediction.
3. **s2.6.a External Data** — "You may use data other than the Competition Data ('External Data') to develop and test your Submissions" if publicly available at no cost or Reasonable. Not engaged: no external data is used.
4. **s2.6.b** — "The use of external data and models is acceptable unless specifically prohibited by the Host." Nothing is prohibited by the Host.
5. **s2.8 Winner's Obligations** — deliver "training code, inference code, and a description of the required computational environment"; s2.5.b: "a detailed description of methodology, where one must be able to reproduce the approach". The pseudo-label step is one script (scripts/l6t_make_pseudo.py) and is reproducible.
6. **Overview › Training Constraints** — "Participants are free to design their algorithms using any approach that complies with the competition rules. Allowed resources include: the provided training data, publicly available datasets (i.e., those indexed in UMUD), publicly available pretrained models. If additional data or pretrained models are used, they must be clearly documented in the final method description."

## Residual risk (recorded, not a prohibition)
The Overview's allowed-resources list ("include") names the training data explicitly and the test images only through s2.4.a. Mitigation: document the pseudo-label step in the s2.8 method description; keep the constant floor L1 as final 2 (L9). Stephen may veto before any L6t-b submission goes out.
