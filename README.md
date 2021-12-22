# DS Pipeline Artifact
This repository contains the source code and data used for the **ICSE 2022** paper -

**Title-** The Art and Practice of Data Science Pipelines: A Comprehensive Study of Data Science Pipelines In Theory, In-The-Small, and In-The-Large

**Authors-** Sumon Biswas, Mohammad Wardat and Hridesh Rajan

**PDF-** https://arxiv.org/abs/2112.01590

## Index
> 1. [Installation](#installation)
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

## Installation

The source code is written in Python 3 and tested using Python 3.7 on Mac OS. It is recommended to use a virtual Python environment for this setup. Furthermore, we used bash shell scripts to automate running benchmark and Python scripts.

Follow the instructions below to set up the environment and run the source code.

Follow these steps to create a virtual environment:

1. Install Anaconda [[installation guide](https://docs.anaconda.com/anaconda/install/)].

2. Create a new Python environment. Run on the command line:
```
conda create --name pipeline python=3.7
conda activate pipeline
```
The shell should look like: `(pipeline) $ `. Now, continue to step 2 and install packages using this shell.
When you are done with the experiments, you can exit this virtual environment by `conda deactivate`.

3. Clone this repository.
```
git clone https://github.com/anonymous-authorss/DS-Pipeline.git
```

4. Navigate to the cloned repository: `cd DS-Pipeline/` and install required packages:
```
pip install -r requirements.txt
```

### Run the pipeline generator tool
Navigate to the benchmark directory `cd src/`.
To run the pipeline generator run
```
python pipeline-generator.py
```
The tool takes two inputs:
1. The notebooks that it parses as `.py` files.
2. The API dictionary that maps each API to a pipeline stage.

The tool produces one output file `pipe.txt` that contains generated pipelines. The output is organized as follows: the name of the root directory is followed by the python scripts under the directory, and then it prints the pipeline. The stages are: Data **A**cquisition, Data **Pr**eparation, **M**odeling, **Tr**aining, **E**valuation, **P**rediction.


## DS Pipelines in Theory

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

**All the data used in analyzing pipelines in-the-small are shared in [this directory](data/small).**
* [This spreadsheet](data/small/all-kaggle-notebooks-url.xlsx) contains the URLs of the 105 Kaggle notebooks used in the analysis.
* The API dictionary used to infer the stages is shared in [this spreadsheet](data/small/API-dictionary.xlsx).
* Details of the high-level pipelines found in 34 Kaggle notebooks are shared in [this spreadsheet](data/small/high-level-pipeline-kaggle.xlsx).

**The source code of the 105 Kaggle notebooks is shared in [this directory](notebooks/).** Kaggle categorized the pipelines into four: analytics, featured, recruitment, and research.

**Static analysis to generate low-level pipelines using the API dictionary are shared in this directory.**
The API dictionary used by the tool is stored in the `stages.csv`. The pipeline generator is written in the `pipeline-generator.py`. In order to run the generator on the Kaggle notebooks, follow the [installation instructions](#installation).

## DS Pipelines In-The-Large
To analyze pipelines in-the-large, we selected GitHub projects from a curated list of data science repositories. All the details of the projects are shared in [this spreadsheet](data/large/GitHub-projects.xlsx). There are three sheets in the spreadsheet:

* The project name, its purpose, number of contributors, AST count of the project, number of source files are stored in the first sheet.
* From the curated benchmark of repositories, initially we selected 269 matured projects. Those URLs of those projects are shared in the second sheet.
* Finally, to further analyze the characteristics, we extracted the language used in the projects, dependencies, etc., which are stored in the third sheet.


## Cite the paper as

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
