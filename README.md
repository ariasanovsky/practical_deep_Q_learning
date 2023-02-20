# practical_deep_Q_learning

## setup

 Practicing PyToch and deep Q learning with [Phil Taber's course](https://www.udemy.com/course/deep-q-learning-from-paper-to-code/).  

I recommend setting up the build environment in a Conda terminal with:

```conda env create -f envrionment.yml```

Activate the environment with:

```conda activate qlearn```

For the Atari projects, I had to install the deprecated package with some combination of the following commands:

```conda install -c conda-forge atari_py```

```conda install "gym[atari,accept-rom-license]"```

```pip install "gym[atari,accept-rom-license]"```

```conda install atari-py```

```pip install atari-py```

I also downloaded the ROMs suggested by OpenAI.  The instructions are linked in the terminal.  Next up, I ran:

```python -m atari_py.import_roms <path where ROMs downloaded>```

and that seemed sufficient!  At a later time, I'll work on finding the cleanest setup for the environment.  