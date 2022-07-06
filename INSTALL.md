# Installation and Usage

The source code is written in Python 3 and tested using Python 3.7 on Mac OS and Ubuntu virtual machine (ICSE'22). It is recommended to use a virtual Python environment for this setup. Furthermore, we used bash shell scripts to automate running benchmark and Python scripts.

### Environment Setup
Follow the instructions below to set up the environment and run the source code.

Follow these steps to create a virtual environment:

1. Install Anaconda [[installation guide](https://docs.anaconda.com/anaconda/install/)]. For new installation, run `conda init <SHELL>`. Similiarly, you can also use miniconda (https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links).

2. Create a new Python environment. Run on the command line:
```
conda create --name pipeline python=3.7
```
```
conda activate pipeline
```
The shell should look like: `(pipeline) $ `. Now, continue to step 2 and install packages using this shell.
When you are done with the experiments, you can exit this virtual environment by `conda deactivate`.

3. Clone this repository.
```
git clone https://github.com/sumonbis/DS-Pipeline.git
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
