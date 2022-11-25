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

**This is the repository for the papers:<br>
[History Compression via Language Models in Reinforcement Learning](https://arxiv.org/abs/2205.12258)** and <br>
**[Toward Semantic History Compression for Reinforcement Learning](https://openreview.net/forum?id=97C6klf5shp).**

**You can find a detailed blog post on HELM at [this link](https://ml-jku.github.io/blog/2022/helm/).**

## HELM

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
By default a Tensorboard log is created.

## HELMv2

We have added support for HELMv2 [here](trainers/helmv2_trainer.py).
We have updated the dependencies for HELMv2 in the [environment file](environment.yml).
To run HELMv2, simply update your existing environment, or create a new one. 
        
Afterwards simply call 

    python main.py --var model=HELMv2 --var env=MiniWorld-Sign-v0
    
to run experiments on the 3D MiniWorld environments.
You can find a comprehensive list of MiniWorld environments [here](https://github.com/Farama-Foundation/Miniworld/blob/master/docs/environments.md).
If you encounter problems for training on MiniWorld (NoSuchDisplayException)

    xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python main.py --var model=HELMv2 --var env=MiniWorld-Sign-v0

More details regarding troubleshooting can be found [here](https://github.com/Farama-Foundation/Miniworld/blob/master/docs/troubleshooting.md).
Code for training the semantic mappings from observation to language space, as well as pretrained mappings will be added soon!

## LICENSE
MIT LICENSE

---

If you find our papers and code useful, please consider citing HELM,

    @InProceedings{paischer2022history,
      title = 	 {History Compression via Language Models in Reinforcement Learning},
      author =       {Paischer, Fabian and Adler, Thomas and Patil, Vihang and Bitto-Nemling, Angela and Holzleitner, Markus and Lehner, Sebastian and Eghbal-Zadeh, Hamid and Hochreiter, Sepp},
      booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
      pages = 	 {17156--17185},
      year = 	 {2022},
      editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
      volume = 	 {162},
      series = 	 {Proceedings of Machine Learning Research},
      month = 	 {17--23 Jul},
      publisher =    {PMLR},
      pdf = 	 {https://proceedings.mlr.press/v162/paischer22a/paischer22a.pdf},
      url = 	 {https://proceedings.mlr.press/v162/paischer22a.html},
      abstract = 	 {In a partially observable Markov decision process (POMDP), an agent typically uses a representation of the past to approximate the underlying MDP. We propose to utilize a frozen Pretrained Language Transformer (PLT) for history representation and compression to improve sample efficiency. To avoid training of the Transformer, we introduce FrozenHopfield, which automatically associates observations with pretrained token embeddings. To form these associations, a modern Hopfield network stores these token embeddings, which are retrieved by queries that are obtained by a random but fixed projection of observations. Our new method, HELM, enables actor-critic network architectures that contain a pretrained language Transformer for history representation as a memory module. Since a representation of the past need not be learned, HELM is much more sample efficient than competitors. On Minigrid and Procgen environments HELM achieves new state-of-the-art results. Our code is available at https://github.com/ml-jku/helm.}
    }
    
and HELMv2:
    
    @inproceedings{paischer2022toward,
        title={Toward Semantic History Compression for Reinforcement Learning},
        author={Fabian Paischer and Thomas Adler and Andreas Radler and Markus Hofmarcher and Sepp Hochreiter},
        booktitle={Second Workshop on Language and Reinforcement Learning},
        year={2022},
        url={https://openreview.net/forum?id=97C6klf5shp}
    }

