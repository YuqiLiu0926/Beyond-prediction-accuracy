# GitHub and archival release procedure

This project is distributed as two linked records:

1. `YuqiLiu0926/Beyond-prediction-accuracy` contains source code, tests and
   reproducibility documentation. Maintain the code on GitHub and archive a
   tagged release through Zenodo.
2. `neural-pde-decision-reliability-data` contains datasets, model weights,
   decision archives, closed-loop records and Source Data. Deposit this
   directory as a separate Zenodo dataset record because its size is not
   suitable for a normal Git repository.

Nature Machine Intelligence requires the code central to the conclusions to be
available to editors and reviewers on request. Upon publication, Nature
Portfolio recommends a DOI-minting repository for custom code. The minimum data
needed to interpret, verify and extend the article must also have a transparent
access route.

Official guidance:

- Nature Machine Intelligence reporting and availability policy:
  https://www.nature.com/natmachintell/editorial-policies/reporting-standards
- GitHub repository creation:
  https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- GitHub large-file limits:
  https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- Zenodo GitHub integration:
  https://help.zenodo.org/docs/github/
- Zenodo dataset upload and DOI reservation:
  https://help.zenodo.org/docs/deposit/create-new-upload/

## 1. Resolve release metadata

Before publication:

- select a code license and replace `LICENSE_SELECTION_REQUIRED.md` with the
  corresponding `LICENSE` file;
- confirm that the companion data package retains its approved CC BY 4.0
  license;
- complete all author names and ORCIDs in `CITATION.cff`;
- replace repository and DOI placeholders in the manuscript only after the
  records exist;
- rebuild and verify both SHA-256 manifests.

MIT or BSD-3-Clause are common code licenses. Code-license selection requires
approval from all rights holders and the authors' institution. The companion
dataset uses CC BY 4.0.

## 2. Publish the code repository

The canonical repository is
https://github.com/YuqiLiu0926/Beyond-prediction-accuracy. Clone it once, then
use normal reviewed commits for subsequent updates:

```powershell
git clone https://github.com/YuqiLiu0926/Beyond-prediction-accuracy.git
cd Beyond-prediction-accuracy
git pull --ff-only
git add .
git status
git commit -m "Update public manuscript code"
git push origin main
```

Inspect `git status` before committing. The `external_data` link, generated
outputs, caches and local environments must remain untracked.

## 3. Archive a software release

Connect the GitHub account to Zenodo, enable the repository, and create a tagged
GitHub release:

```powershell
git tag -a v1.0.0 -m "Manuscript release v1.0.0"
git push origin v1.0.0
```

Create the corresponding GitHub release from tag `v1.0.0`. Zenodo will ingest
the enabled release and assign a version-specific software DOI. Check the
Zenodo record metadata before using the DOI in the manuscript. `CITATION.cff`
is sufficient unless Zenodo-specific metadata is required.

## 4. Deposit the data and weights

Create a separate Zenodo dataset draft. Upload the contents of
`neural-pde-decision-reliability-data` as one or more archives while retaining
the top-level directory structure. Complete the metadata using
`zenodo_metadata.json.template`, reserve a DOI, and validate the upload against
`FILE_MANIFEST.csv` before publication.

After rebuilding and verifying both manifests, create deterministic upload
archives and their SHA-256 checksums with:

```powershell
cd <path-to-code>/neural-pde-decision-reliability
python tools/create_release_archives.py --version v1.0.0
```

The command writes a small software archive, the complete companion data
archive and `SHA256SUMS.txt` to the sibling `release_upload` directory. Upload
the data archive to the Zenodo dataset record. The software archive is a local
submission copy; the version of record should be the tagged GitHub release
archived by Zenodo.

The dataset record should include:

- four trajectory datasets;
- thirteen frozen model checkpoints;
- the 36 decision archives;
- critical-event, operator-defect and recoverability records;
- Gray--Scott and thermal closed-loop records;
- numerical-level qualification records;
- Source Data workbooks and machine-readable CSV exports;
- README, data dictionary, manifest and license.

Do not put the 2.4 GB data package in ordinary Git history. GitHub blocks files
larger than 100 MiB and recommends keeping repositories small.

## 5. Peer-review access

If public release must wait until acceptance, keep the GitHub repository private
and provide editor and reviewer access through the journal's permitted review
mechanism. A restricted Zenodo record can provide a review link for the data.
Disclose any temporary restriction in the submitted Data and Code Availability
statements. Make the final records public promptly upon publication.

## 6. Final manuscript statements

The final Data Availability statement should name the deposited dataset,
repository and full DOI URL. Cite the dataset formally in the reference list.
The Code Availability statement should provide the GitHub URL and archived
software DOI. Use the exact tagged release corresponding to the submitted
results.
