# AD-AutoGPT: An Autonomous GPT for Alzheimer's Disease Infodemiology
\[In submission\] Code for: [AD-AutoGPT: An Autonomous GPT for Alzheimer's Disease Infodemiology](https://arxiv.org/abs/2306.10095)

In this pioneering study, inspired by AutoGPT, the state-of-the-art open-source application based on the GPT-4 large language model, we develop a novel tool called AD-AutoGPT which can conduct data collection, processing, and analysis about complex health narratives of Alzheimer's Disease in an autonomous manner via users' textual prompts. We collated comprehensive data from a variety of news sources, including the Alzheimer's Association, BBC, Mayo Clinic, and the National Institute on Aging since June 2022, leading to the autonomous execution of robust trend analyses, intertopic distance maps visualization, and identification of salient terms pertinent to Alzheimer's Disease. This approach has yielded not only a quantifiable metric of relevant discourse but also valuable insights into public focus on Alzheimer's Disease. This application of AD-AutoGPT in public health signifies the transformative potential of AI in facilitating a data-rich understanding of complex health narratives like Alzheimer's Disease in an autonomous manner, setting the groundwork for future AI-driven investigations in global health landscapes.


Our framework is as follows.

![pipeline](./f1/framework.jpg)


## :hammer_and_wrench: Requirements

```python3
git clone https://github.com/levyisthebest/AD-AutoGPT.git
pip install -r requirements.txt
```

## Dataset 
All data is public data on the network, and the data automatically collected by the program will be saved in the workplace directory. The program itself contains the data used in this paper. The news collected from 4 websites can be seen in ![workplace](./workplace) folder.

## :gear: How to run

 First, you need to set up the environment, then you can start using AD-AutoGPT. The main code of AD-AutoGPT contains two parts, AD_GPT.py and AD_GPT_tools.py. AD_GPY.py is the main code of the AD_AutoGPT, it will set the basic structure of code and call specific functions from AD_GPT_tools.py to solve the specific problems. The other files in this repository contains the system file(dictionary.gensim,model3.gensim..) to help extract city name from the original news. world_map.shp and world_map.shx are the files to draw the world map. More details can be seen in the comments of AD_GPT.py and AD_GPT_tools.py. 


To run the AD_AutoGPT:

 1. Set OPENAI_API_KEY environment variable to your OpenAI API key:
```
export OPENAI_API_KEY=<your key>
```

2. Run the script to start AD-AutoGPT
```
.\run.bat ## or sh run.sh
```
3. To keep the tools up to date and include more current articles, you can input the following prompt through the command line:
```
Can you help me find the up-to-date news for Alzheimer Disease and help me understand them?
```
Then, AD-AutoGPT will autonomously scratch the newest information of AD and save important informations and the visulization results in the ![workplace](./workplace) folder. 


## Citation
If you use the code, please cite the following paper:
```
@article{dai2023ad,
  title={AD-AutoGPT: An Autonomous GPT for Alzheimer's Disease Infodemiology},
  author={Dai, Haixing and Li, Yiwei and Liu, Zhengliang and Zhao, Lin and Wu, Zihao and Song, Suhang and Shen, Ye and Zhu, Dajiang and Li, Xiang and Li, Sheng and others},
  journal={arXiv preprint arXiv:2306.10095},
  year={2023}
}
```
