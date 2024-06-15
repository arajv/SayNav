# SayNav
Grounding Large Language Models for Dynamic Planning to Navigation in New Environments

## About
We present SayNav, a new approach that leverages human knowledge from Large Language Models (LLMs) for efficient generalization to complex navigation tasks in unknown large-scale environments. SayNav uses a novel grounding mechanism, that incrementally builds a 3D hierarchical scene graph of the explored environment as inputs to LLMs, for generating feasible and contextually appropriate high-level plans for navigation. The LLM-generated plan is then executed by a pre-trained low-level planner, that treats each planned step as a short-distance point-goal navigation sub-task.

## Project Page
Visit our [Project Page](https://www.sri.com/ics/computer-vision/saynav-grounding-large-language-models-for-dynamic-planning-to-navigation-in-new-environments/) to look at various demonstration videos.

## Citing SayNav
If you find our work useful, please cite our [paper](https://ojs.aaai.org/index.php/ICAPS/article/view/31506/33666).
```
@inproceedings{arajv2024Saynav,
  title     =     {SayNav: Grounding Large Language Models for Dynamic Planning to Navigation in New Environments},
  author    =     {Abhinav Rajvanshi and Karan Sikka and Xiao Lin and Bhoram Lee and Han-Pang Chiu and Alvaro Velasquez},
  booktitle =     {International Conference on Automated Planning and Scheduling (ICAPS)},
  year      =     {2024}
}
```

## Installation
### Install Vulkan Library
```sh
sudo apt-get install -y libvulkan1 vulkan-utils
```
### Setup a Conda Environment
Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda environment:
```
conda create -n saynav python=3.9
conda activate saynav
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda==11.7 -c pytorch -c nvidia
pip install -r requirements.txt
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

## Running the pipeline
We provide a run script which runs the typical SayNav pipeline on a house instance from ProcTHOR dataset with 3 random objects.
You will need to add your OpenAI API key in run/test.py.
```
export PYTHONPATH={Path to Repository}/src/cython
cd {Path to Repository}/run
python test.py
```
