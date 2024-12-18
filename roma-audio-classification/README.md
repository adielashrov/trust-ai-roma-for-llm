# RoMA - Audio - Classification

The following instrcutions are for installing and executing the roma-audio-classification case study on a linux/ubuntu machine with a GPU properly configured.

### Setting up the environement
1. Start by making sure that you have the following packages installed:

`sudo apt-get install python3.9-dev`

`sudo  apt-get  install libasound2-dev`

2. Create a virtual environment using conda:

`conda create -n roma-audio python=3.9`

3. Activate the environment:

`conda activate roma-audio`

5. Install the requirements file

`pip install -r requirements.txt`

### Executing the experiment

`python main.py`

Good Luck!
