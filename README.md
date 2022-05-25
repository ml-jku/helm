# History Compression via Language Models in Reinforcement Learning

Fabian Paischer<sup>1 2</sup>,
Thomas Adler<sup>1</sup>,
Vihang Patil<sup>1</sup>,
Angela Bitto-Nemling<sup>1 3</sup>,
Markus Holzleitner<sup>1</sup>,
Sebastian Lehner<sup>1 2</sup>,
Hamid Eghbal-zadeh<sup>1</sup>,
Sepp Hochreiter<sup>1 2 3</sup>

<sup>1</sup> LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria</br>
<sup>2</sup> ELLIS Unit Linz  
<sup>3</sup> Institute of Advanced Research in Artificial Intelligence (IARAI)

---

**This is the repository for the paper:
[History Compression via Language Models in Reinforcement Learning](https://arxiv.org/abs/2205.12258).**

**Detailed blog post on this paper at [this link](https://ml-jku.github.io/blog/2022/helm/).**

---

To reproduce our results, first clone the repository and install the conda environment by

    git clone https://github.com/ml-jku/helm.git
    cd helm
    conda env create -f environment.yml

After installing the conda environment you can train HELM on the KeyCorridor environment by

    python main.py

A new directory `./experiments/HELM/MiniGrid-KeyCorridorS3R1-v0` will be created in which all log files and checkpoints will be stored.

All changeable parameters are stored in the `config.json` file and can be adjusted via command line arguments as:

    python main.py --var KEY=VALUE

For example, if you would like to train on `RandomMaze-v0`:

    python main.py --var env=RandomMaze-v0

or on the Procgen environment `maze`:

    python main.py --var env=maze

**Note** that by default the Procgen environments are created in the *memory* distribution mode, thus only the six environments 
as mentioned in the paper can be trained on, all others do not support the *memory* mode.

By default a Tensorboard log is created which can be visualized by

    tensorboard --logdir ./experiments

## LICENSE
MIT LICENSE
