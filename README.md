# DS Pipeline Artifact
This repository contains the source code and data used for the **ICSE 2022** paper -

#### The Art and Practice of Data Science Pipelines: A Comprehensive Study of Data Science Pipelines In Theory, In-The-Small, and In-The-Large

**Authors** Sumon Biswas, Mohammad Wardat and Hridesh Rajan
**PDF** https://arxiv.org/abs/2112.01590

## Index
> 1. [Installation](/INSTALL.md)
> 2. [DS pipelines in theory](#ds-pipelines-in-theory)
  >> * All [labeled pipelines](pipelines.pdf) from art and practice.
  >> * [Data](data/theory): Raw pipelines, labels from raters, frequency of stages
> 3. [DS pipelines in-the-small](#ds-pipelines-in-the-small)
  >> * [Data](data/small): List of Kaggle notebooks, API dictionary, high-level Kaggle pipelines
  >> * [Kaggle notebooks code](notebooks/)
  >> * Source code to [generate low-level pipeline](src/)
> 4. [DS pipelines in-the-large](#ds-pipelines-in-the-large)
  >> * [Data](data/large): Github projects and details

![Representative Data Science Pipeline](/pipeline.jpg)

**The Artifact:** This artifact is divided into the following three main sections. Only the *DS pipelines in-the-small* contains software that requires installation. The other two sections contains additional and detailed data used in the paper. We also published the artifact in Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5866584.svg)](https://doi.org/10.5281/zenodo.5866584)


## DS Pipelines in Theory

In this section, we conducted a survey of DS pipelines from literature and popular press. After collecting the pipelines, we have adopted a open-coding scheme to label the pipelines and propose representative stages. The artifact contains all the references of the literature review, collected raw pipelines, and the labels from the raters.

**The 71 pipelines in theory with the labels and references can be found [here](pipelines.pdf).**

* Each row is a pipeline and the columns are stages or whether the pipeline involves cyber/physical/human components.
* The pipelines are categorized into three: 1) Machine learning process, 2) Big data, and 3) Team Process.
* The pipelines are color-coded based on the overall goal of the article: Describe/propose, Survey/compare/review, Data science optimization, and Introduce new application.

**The labels from both the raters and the reconciled labels are assembled in [this spreadsheet](data/theory/all-labels.xlsx).**
The spreadsheet contains three sheets - for rater 1, rater 2, and reconciled ones. The rows are the pipelines and columns are the stages.

**The raw pipelines collected from the above references can be found in [this spreadsheet](data/theory/collected-raw-pipelines.xlsx).**
We extracted the raw pipelines from each of the selected 71 articles. The spreadsheet contains the description of the pipeline, screenshot, and URL.

**Frequency of the stages are calculated in [this spreadsheet](data/theory/frequency-of-stages.xlsx).**
From the labeling of the pipelines, we calculated how many times each stage appeared.

## DS Pipelines In-The-Small

This section contains both the data and software. We collected Kaggle notebooks and built a static analysis tool to automatically extract pipelines. The descriptions of the necessary data and instructions to run the tools is given below.

**All the data used in analyzing pipelines in-the-small are shared in [this directory](data/small).**
* [This spreadsheet](data/small/all-kaggle-notebooks-url.xlsx) contains the URLs of the 105 Kaggle notebooks used in the analysis.
* The API dictionary used to infer the stages is shared in [this spreadsheet](data/small/API-dictionary.xlsx).
* Details of the high-level pipelines found in 34 Kaggle notebooks are shared in [this spreadsheet](data/small/high-level-pipeline-kaggle.xlsx).

**The source code of the 105 Kaggle notebooks is shared in [this directory](notebooks/).** Kaggle categorized the pipelines into four: analytics, featured, recruitment, and research.

**Static analysis to generate low-level pipelines using the API dictionary are shared in this directory.**
The API dictionary used by the tool is stored in the `stages.csv`. The pipeline generator is written in the `pipeline-generator.py`. In order to run the generator on the Kaggle notebooks, follow the [installation instructions](/INSTALL.md).

## DS Pipelines In-The-Large

To analyze pipelines in-the-large, we selected GitHub projects from a curated list of data science repositories. All the details of the projects are shared in [this spreadsheet](data/large/GitHub-projects.xlsx). There are three sheets in the spreadsheet:

* The project name, its purpose, number of contributors, AST count of the project, number of source files are stored in the first sheet.
* From the curated benchmark of repositories, initially we selected 269 matured projects. Those URLs of those projects are shared in the second sheet.
* Finally, to further analyze the characteristics, we extracted the language used in the projects, dependencies, etc., which are stored in the third sheet.

### Contact

1. Sumon Biswas, Iowa State University (sumon@iastate.edu)
2. Mohammad Wardat, Iowa State University (wardat@iastate.edu)
3. Hridesh Rajan, Iowa State University (hridesh@iastate.edu)

### Cite the paper as

```
@inproceedings{biswas22art,
  author = {Sumon Biswas and Mohammad Wardat and Hridesh Rajan},
  title = {The Art and Practice of Data Science Pipelines: A Comprehensive Study of Data Science Pipelines In Theory, In-The-Small, and In-The-Large},
  booktitle = {ICSE'22: The 44th International Conference on Software Engineering},
  location = {Pittsburgh, PA, USA},
  month = {May 21-May 29},
  year = {2022},
  entrysubtype = {conference}
}
```
