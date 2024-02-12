# XRL 


## Useful resources

[1] [Explainable Reinforcement Learning (XRL) Resources](https://github.com/yanzheb/XRL) by Yanzhe Bekkemoen


## XRL for Graph Machine Learning

[1] Saha, Sayan, Monidipa Das, and Sanghamitra Bandyopadhyay. ["A Model-Centric Explainer for Graph Neural Network based Node Classification."](https://dl.acm.org/doi/abs/10.1145/3511808.3557535) Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022. [Github](https://github.com/KingGuzman/Node-Classifier-Explainer)

[2] Mishra, Saurabh, and Sonia Khetarpaul. ["Predicting Taxi Hotspots in Dynamic Conditions Using Graph Neural Network."](https://link.springer.com/chapter/10.1007/978-3-031-15512-3_7) Australasian Database Conference. Cham: Springer International Publishing, 2022.

[3] Bacciu, Davide, and Danilo Numeroso. ["Explaining deep graph networks via input perturbation."](https://ieeexplore.ieee.org/abstract/document/9761788/) IEEE Transactions on Neural Networks and Learning Systems (2022). [Github](https://github.com/danilonumeroso/legit)

[4] Jin, Jiarui, et al. ["Graph-Enhanced Exploration for Goal-oriented Reinforcement Learning."](https://openreview.net/forum?id=rlYiXFdSy70) (2021).

[5] Peng, Hao, et al. ["Reinforced neighborhood selection guided multi-relational graph neural networks."](https://dl.acm.org/doi/abs/10.1145/3490181) ACM Transactions on Information Systems (TOIS) 40.4 (2021): 1-46. [Github](https://github.com/safe-graph/RioGNN)

[6] Wickman, Ryan, Xiaofei Zhang, and Weizi Li. ["A Generic Graph Sparsification Framework using Deep Reinforcement Learning."](https://arxiv.org/abs/2112.01565) arXiv preprint arXiv:2112.01565 (2021). [Github](https://github.com/rwickman/SparRL-PyTorch)

[5] Yuan, Hao, et al. ["On explainability of graph neural networks via subgraph explorations."](http://proceedings.mlr.press/v139/yuan21c.html) International conference on machine learning. PMLR, 2021. [Github](https://github.com/divelab/DIG)

[6] Yuan, Hao, et al. ["Xgnn: Towards model-level explanations of graph neural networks."](https://dl.acm.org/doi/abs/10.1145/3394486.3403085) Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.

[7] Xian, Yikun, et al. ["Reinforcement knowledge graph reasoning for explainable recommendation."](https://dl.acm.org/doi/abs/10.1145/3331184.3331203?casa_token=rH-9rqcddjIAAAAA:HNQM9AflQOpCcSNZfjXMAFCZCEP7bmcHzeFTm3Zyoj5l63ryhmmWgVcliQ5iOJTyfQVY65HCsGpJLzM) Proceedings of the 42nd international ACM SIGIR conference on research and development in information retrieval. 2019.

[8] Song, Weiping, et al. ["Ekar: an explainable method for knowledge aware recommendation."](https://ui.adsabs.harvard.edu/abs/2019arXiv190609506S/abstract) arXiv preprint arXiv:1906.09506 (2019).

[9] Dai, Hanjun, et al. ["Learning transferable graph exploration."](https://proceedings.neurips.cc/paper/2019/hash/afe434653a898da20044041262b3ac74-Abstract.html) Advances in Neural Information Processing Systems 32 (2019).

## Feature Importance (FI) Explanations

FI explanations provide an action-level look at the agent’s behavior. 
- These techniques answer questions: **What immediate context inluenced the agent to perform that action?**

**Convert Policy to Interpretable Format**

[1] Osbert Bastani, Jeevana Priya Inala, and Armando Solar-Lezama. 2022. [Interpretable, Veriiable, and Robust Reinforcement Learning via Program Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-04083-2_11#Sec2). In International Workshop on Extending Explainable AI Beyond Deep Models and Classiiers. Springer, 207-228.

[2] Osbert Bastani, Yewen Pu, and Armando Solar-Lezama. 2018. [Veriiable reinforcement learning via policy extraction](https://arxiv.org/pdf/1805.08328.pdf). In Advances in Neural Information Processing Systems. 2494-2504.

[3] Tom Bewley and Jonathan Lawry. 2020. [Tripletree: A versatile interpretable representation of black box agents and their environments](https://arxiv.org/pdf/2009.04743.pdf). CoRR, abs/2009.04743 (2020).

[4] Yuxin Dai, Qimei Chen, Jun Zhang, Xiaohui Wang, Yilin Chen, Tianlu Gao, Peidong Xu, Siyuan Chen, Siyang Liao, Huaiguang Jiang, et al. 2022. [Enhanced Oblique Decision Tree Enabled Policy Extraction for Deep Reinforcement Learning in Power System Emergency Control](https://www.sciencedirect.com/science/article/abs/pii/S0378779622001626). Electric Power Systems Research 209 (2022), 107932.

[5] Wei Guo and Peng Wei. 2022. Explainable Deep Reinforcement Learning for Aircraft Separation Assurance. 4th Digital Avionics Systems Conference (2022).

[6] Aman Jhunjhunwala. 2019. Policy Extraction via Online Q-Value Distillation. Masters Thesis, University of Waterloo (2019).

[7] Guiliang Liu, Oliver Schulte, Wang Zhu, and Qingcan Li. 2018. Toward interpretable deep reinforcement learning with linear model U-trees. In Proceedings of the Joint European Conference on Machine Learning and Knowledge Discovery in Databases. 414ś429.

[8] Abhinav Verma, Vijayaraghavan Murali, Rishabh Singh, Pushmeet Kohli, and Swarat Chaudhuri. 2018. Programmatically Interpretable Reinforcement Learning. In Proceedings of the 35th International Conference on Machine Learning. 5045ś5054.

[9] Hengzhe Zhang, Aimin Zhou, and Xin Lin. 2020. Interpretable policy derivation for reinforcement learning based on evolutionary feature synthesis. Complex & Intelligent Systems (2020), 1ś13.

**Learn an Intrinsically Interpretable Policy**

[1] Leonardo Lucio Custode and Giovanni Iacca. 2022. Interpretable AI for policy-making in pandemics. arXiv preprint arXiv:2204.04256 (2022).

[2] Leonardo Lucio Custode and Giovanni Iacca. 2022. Interpretable pipelines with evolutionary optimized modules for reinforcement learning tasks with visual inputs. In Proceedings of the Genetic and Evolutionary Computation Conference Companion. 224-227.

[3] Yinglong Dai, Haibin Ouyang, Hong Zheng, Han Long, and Xiaojun Duan. 2022. Interpreting a deep reinforcement learning model with conceptual embedding and performance analysis. Applied Intelligence (2022), 1-17.

[4] Yashesh Dhebar, Kalyanmoy Deb, Subramanya Nageshrao, Ling Zhu, and Dimitar Filev. 2022. Toward Interpretable-AI Policies Using Evolutionary Nonlinear Decision Trees for Discrete-Action Systems. IEEE Transactions on Cybernetics (2022).

[5] Daniel Hein, Alexander Hentschel, Thomas Runkler, and Stefen Udluft. 2017. Particle swarm optimization for generating interpretable fuzzy reinforcement learning policies. Engineering Applications of Artiicial Intelligence 65 (2017), 87 - 98.

[6] Daniel Hein, Stefen Udluft, and Thomas A. Runkler. 2019. Interpretable Policies for Reinforcement Learning by Genetic Programming. In Proceedings of the Genetic and Evolutionary Computation Conference.

[7] Mikel Landajuela, Brenden K Petersen, Sookyung Kim, Claudio P Santiago, Ruben Glatt, Nathan Mundhenk, Jacob F Pettit, and Daniel Faissol. 2021. Discovering symbolic policies with deep reinforcement learning. In International Conference on Machine Learning. PMLR, 5979-5989.

[8] Francis Maes, Raphael Fonteneau, Louis Wehenkel, and Damien Ernst. 2012. Policy search in a space of simple closed-form formulas: Towards interpretability of reinforcement learning. In Proceedings of the 15th International Conference on Discovery Science. 37-51.

[9] Rohan Paleja, Yaru Niu, Andrew Silva, Chace Ritchie, Sugju Choi, and Matthew Gombolay. 2022. Learning Interpretable, High-Performing Policies for Autonomous Driving. Learning (2022), 1.

[10] Ivan Dario Jimenez Rodriguez, Taylor W. Killian, Sung-Hyun Son, and Matthew C. Gombolay. 2019. Optimization Methods for Interpretable Diferentiable Decision Trees in Reinforcement Learning. arXiv preprint, arXiv:1903.09338 (2019).

[11] Zhihao Song, Yunpeng Jiang, Jianyi Zhang, Paul Weng, Dong Li, Wulong Liu, and Jianye Hao. 2022. An Interpretable Deep Reinforcement Learning Approach to Autonomous Driving. IJCAI Workshop on Artiicial Intelligence for Automous Driving (2022).

[12] Nicholay Topin, Stephanie Milani, Fei Fang, and Manuela Veloso. 2021. Iterative Bounding MDPs: Learning Interpretable Policies via Non-Interpretable Methods. arXiv preprint arXiv:2102.13045 (2021).

[13] Varun Ravi Varma. 2021. Interpretable Reinforcement Learning with the Regression Tsetlin Machine. Ph. D. Dissertation.

[14] Li Zhang, Xin Li, Mingzhong Wang, and Andong Tian. 2021. Of-Policy Diferentiable Logic Reinforcement Learning. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 617-632.

**Directly Generate Explanation**

[1] Andrew Anderson, Jonathan Dodge, Amrita Sadarangani, Zoe Juozapaitis, Evan Newman, Jed Irvine, Souti Chattopadhyay, Alan Fern, and Margaret Burnett. 2019. Explaining reinforcement learning to mere mortals: An empirical study. In Proceedings of the 28th International Joint Conference on Artiicial Intelligence.

[2] Raghuram Mandyam Annasamy and Katia Sycara. 2019. Towards better interpretability in deep q-networks. In Proceedings of the 33rd AAAI Conference on Artiicial Intelligence (AAAI-19), Vol. 33. 4561ś4569.

[3] Akanksha Atrey, Kaleigh Clary, and David Jensen. 2020. Exploratory Not Explanatory: Counterfactual Analysis of Saliency Maps for Deep RL. In Proceedings of the 8th International Conference on Learning Representations.

[4] Upol Ehsan, Brent Harrison, Larry Chan, and Mark Riedl. 2018. Rationalization: A Neural Machine Translation Approach to Generating Natural Language Explanations. Proceedings of the 1st AAAI/ACM Conference on Artiicial Intelligence, Ethics, and Society (2018).

[5] Vikash Goel, Jameson Weng, and Pascal Poupart. 2018. Unsupervised video object segmentation for deep reinforcement learning. In Advances in Neural Information Processing Systems. 5683ś5694.

[6] Bradley Hayes and Julie A Shah. 2017. Improving robot controller transparency through autonomous policy explanation. In Proceedings
of the 2017 12th ACM/IEEE International Conference on Human-Robot Interaction (HRI. IEEE, 303-312.)

[7] Tobias Huber, Dominik Schiller, and Elisabeth André. 2019. Enhancing explainability of deep reinforcement learning through selective layer-wise relevance propagation. In Proceedings of the Joint German/Austrian Conference on Artiicial Intelligence (Künstliche Intelligenz). 188-202.

[8] Hidenori Itaya, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi, and Komei Sugiura. 2021. Visual Explanation using Attention Mechanism in Actor-Critic-based Deep Reinforcement Learning. In 2021 International Joint Conference on Neural Networks (IJCNN). IEEE, 1-10.

[9] Rahul Iyer, Yuezhang Li, Huao Li, Michael Lewis, Ramitha Sundar, and Katia Sycara. 2018. Transparency and Explanation in Deep Reinforcement Learning Neural Networks. In Proceedings of the 1st AAAI/ACM Conference on Artiicial Intelligence, Ethics, and Society.

[10] Woo Kyung Kim, Youngseok Lee, and Honguk Woo. 2022. Mean-variance Based Risk-sensitive Reinforcement Learning with Inter- pretable Attention. In 2022 the 5th International Conference on Machine Vision and Applications (ICMVA). 104ś109.

[11] Samuel Greydanus, Anurag Koul, Jonathan Dodge, and Alan Fern. 2018. Visualizing and Understanding Atari Agents. In Proceedings of the 35th International Conference on Machine Learning. 1792ś1801.

[12] Alexander Mott, Daniel Zoran, Mike Chrzanowski, Daan Wierstra, and Danilo Jimenez Rezende. 2019. Towards interpretable reinforcement learning using attention augmented agents. In Advances in Neural Information Processing Systems. 12329ś12338.

[13] Matthew L Olson, Roli Khanna, Lawrence Neal, Fuxin Li, and Weng-Keen Wong. 2021. Counterfactual state explanations for reinforcement learning agents via generative deep learning. Artiicial Intelligence 295 (2021), 103455.

[14] Michele Persiani and Thomas Hellström. 2022. The Mirror Agent Model: a Bayesian Architecture for Interpretable Agent Behavior. In 4th International Workshop on EXplainable and TRAnsparent AI and Multi-Agent Systems (EXTRAAMAS 2022), Online via Auckland, NZ, May 9-10, 2022.

[15] Christian Rupprecht, Cyril Ibrahim, and Christopher J Pal. 2020. Finding and Visualizing Weaknesses of Deep Reinforcement Learning Agents. In Proceedings of the 8th International Conference on Learning Representations.

[16] Wenjie Shi, Zhuoyuan Wang, Shiji Song, and Gao Huang. 2020. Self-Supervised Discovering of Causal Features: Towards Interpretable Reinforcement Learning. arXiv preprint, arXiv:2003.07069 (2020).

[17] Yujin Tang, Duong Nguyen, and David Ha. 2020. Neuroevolution of Self-Interpretable Agents. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference.

[18] Stephan Wäldchen, Sebastian Pokutta, and Felix Huber. 2022. Training characteristic functions with reinforcement learning: Xai-methods play connect four. In International Conference on Machine Learning. PMLR, 22457ś22474.

[19] Xinzhi Wang, Schengcheng Yuan, Hui Zhang, Michael Lewis, and Katia Sycara. 2019. Verbal Explanations for Deep Reinforcement
Learning Neural Networks with Attention on Extracted Features. In Proceedings of the 28th IEEE International Conference on Robot and
Human Interactive Communication (RO-MAN).

[20] Laurens Weitkamp, Elise van der Pol, and Zeynep Akata. 2018. Visual rationalizations in deep reinforcement learning for Atari games.
In Proceedings of the 30th Benelux Conference on Artiicial Intelligence. 151ś165.

[21] Qiyuan Zhang, Xiaoteng Ma, Yiqin Yang, Chenghao Li, Jun Yang, Yu Liu, and Bin Liang. 2021. Learning to Discover Task-Relevant Features for Interpretable Reinforcement Learning. IEEE Robotics and Automation Letters 6, 4 (2021), 6601ś6607.


## Learning Process or MDP (LPM) Explanations

LPM explanations provide additional information about the efects of the training process or the MDP.
- These techniques answer questions:
    - **Which training points were most inluential on the agent’s learned behavior?**
    - **Which objectives is the agent prioritizing?**

**Model Domain Information**

[1] Jianyu Chen, Shengbo Eben Li, and Masayoshi Tomizuka. 2021. Interpretable end-to-end urban autonomous driving with latent deep reinforcement learning. IEEE Transactions on Intelligent Transportation Systems (2021).

[2] Francisco Cruz, Richard Dazeley, and Peter Vamplew. 2019. Memory-Based Explainable Reinforcement Learning. AI 2019: Advances in Artiicial Intelligence (2019).

[3] Francisco Cruz, Richard Dazeley, and Peter Vamplew. 2020. Explainable robotic systems: Understanding goal-driven actions in a reinforcement learning scenario. arXiv e-prints (2020), arXivś2006.

[4] Yixin Lin, Austin S Wang, Eric Undersander, and Akshara Rai. 2022. Eicient and interpretable robot manipulation with graph neural networks. IEEE Robotics and Automation Letters 7, 2 (2022), 2740ś2747.

[5] Zhengxian Lin, Kim-Ho Lam, and Alan Fern. 2020. Contrastive explanations for reinforcement learning via embedded self predictions.
arXiv preprint arXiv:2010.05180 (2020).

[6] Prashan Madumal, Tim Miller, Liz Sonenberg, and Frank Vetere. 2020. Explainable reinforcement learning through a causal lens. In Proceedings of the 34th AAAI Conference on Artiicial Intelligence (AAAI-20).

[7] Sergei Volodin. 2021. CauseOccam: Learning Interpretable Abstract Representations in Reinforcement Learning Environments via Model Sparsity. Technical Report.

[8] Herman Yau, Chris Russell, and Simon Hadield. 2020. What did you think would happen? explaining agent behaviour through intended outcomes. Advances in Neural Information Processing Systems 33 (2020), 18375ś18386.

**Decompose Reward Function**

[1] Andrew Anderson, Jonathan Dodge, Amrita Sadarangani, Zoe Juozapaitis, Evan Newman, Jed Irvine, Souti Chattopadhyay, Alan
Fern, and Margaret Burnett. 2019. Explaining reinforcement learning to mere mortals: An empirical study. In Proceedings of the 28th
International Joint Conference on Artiicial Intelligence.

[2] Ioana Bica, Daniel Jarrett, Alihan Hüyük, and Mihaela van der Schaar. 2021. Learning "What-if" Explanations for Sequential Decision-Making. In Proceedings of the 9th International Conference on Learning Representations.

[3] Benjamin Beyret, Ali Shafti, and A Aldo Faisal. 2019. Dot-to-Dot: Explainable Hierarchical Reinforcement Learning for Robotic Manipulation. In Proceedings of the 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

[4] Wenbo Guo, Xian Wu, Usmann Khan, and Xinyu Xing. 2021. EDGE: Explaining Deep Reinforcement Learning Policies. Advances in
Neural Information Processing Systems 34 (2021).

[5] Erik Jenner and Adam Gleave. 2022. Preprocessing Reward Functions for Interpretability. arXiv preprint arXiv:2203.13553 (2022).

[6] Zoe Juozapaitis, Anurag Koul, Alan Fern, Martin Erwig, and Finale Doshi-Velez. 2019. Explainable reinforcement learning via reward
decomposition. In Proceedings of the 28th International Joint Conference on Artiicial Intelligence Workshop on Explainable Artiicial Intelligence

[7] Finn Rietz, Sven Magg, Fredrik Heintz, Todor Stoyanov, Stefan Wermter, and Johannes A Stork. 2022. Hierarchical goals contextualize local reward decomposition explanations. Neural Computing and Applications (2022), 1ś12.

**Identify Training Points**

[1] Giang Dao, Indrajeet Mishra, and Minwoo Lee. 2018. Deep Reinforcement Learning Monitor for Snapshot Recording. In Proceedings of
the 2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 591ś598.

[2] Omer Gottesman, Joseph Futoma, Yao Liu, Sonali Parbhoo, Leo Anthony Celi, Emma Brunskill, and Finale Doshi-Velez. 2020. Inter-
pretable Of-Policy Evaluation in Reinforcement Learning by Highlighting Inluential Transitions. arXiv preprint, arXiv:2002.03478
(2020).


## Policy-Level (PL) Explanations

PL explanations present summaries of long-term behavior through abstraction or representative examples.
- These techniques answer questions:
    - **How will the agent behave over time?**

**Summarize Using Transitions**

[1] Dan Amir and Ofra Amir. 2018. Highlights: Summarizing agent behavior to people. In Proceedings of the 17th International Conference
on Autonomous Agents and Multiagent Systems. International Foundation for Autonomous Agents and Multiagent Systems, 1168ś1176.

[2] Marius-Constantin Dinu, Markus Hofmarcher, Vihang P Patil, Matthias Dorfer, Patrick M Blies, Johannes Brandstetter, Jose A Arjona- Medina, and Sepp Hochreiter. 2022. XAI and Strategy Extraction via Reward Redistribution. In International Workshop on Extending Explainable AI Beyond Deep Models and Classiiers. Springer, 177ś205.

[3] Julius Frost, Olivia Watkins, Eric Weiner, Pieter Abbeel, Trevor Darrell, Bryan Plummer, and Kate Saenko. 2022. Explaining Reinforcement Learning Policies through Counterfactual Trajectories. arXiv preprint arXiv:2201.12462 (2022).

[4] Sandy H Huang, Kush Bhatia, Pieter Abbeel, and Anca D Dragan. 2018. Establishing appropriate trust via critical states. In Proceedings
of the 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 3929-3936.

[5] Alexis Jacq, Johan Ferret, Olivier Pietquin, and Matthieu Geist. 2022. Lazy-MDPs: Towards Interpretable Reinforcement Learning by
Learning When to Act. arXiv preprint arXiv:2203.08542 (2022).

[6] Isaac Lage, Daphna Lifschitz, Finale Doshi-Velez, and Ofra Amir. 2019. Exploring computational user models for agent policy summarization. In Proceedings of the 28th International Joint Conference on Artiicial Intelligence.

**Convert RNN to Interpretable Format**

[1] Mohamad H Danesh, Anurag Koul, Alan Fern, and Saeed Khorram. 2021. Re-understanding inite-state representations of recurrent policy networks. In International Conference on Machine Learning. PMLR, 2388-2397.

[2] Mohammadhosein Hasanbeig, Natasha Yogananda Jeppu, Alessandro Abate, Tom Melham, and Daniel Kroening. 2021. DeepSynth: Automata synthesis for automatic task segmentation in deep reinforcement learning. In The Thirty-Fifth AAAI Conference on Artiicial Intelligence, AAAI, Vol. 2. 36.

[3] Anurag Koul, Sam Greydanus, and Alan Fern. 2018. Learning inite state representations of recurrent policy networks. arXiv preprint, arXiv:1811.12530 (2018).

**Extract Clusters or Abstract States**

[1] Joe McCalmon, Thai Le, Sarra Alqahtani, and Dongwon Lee. 2022. CAPS: Comprehensible Abstract Policy Summaries for Explaining Reinforcement Learning Agents. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems. 889-897.

[2] Sarath Sreedharan, Siddharth Srivastava, and Subbarao Kambhampati. 2020. TLdR: Policy Summarization for Factored SSP Problems
Using Temporal Abstractions. Proceedings of the 30th International Conference on Automated Planning and Scheduling.

[3] Nicholay Topin and Manuela Veloso. 2019. Generation of Policy-Level Explanations for Reinforcement Learning. In Proceedings of the 33rd AAAI Conference on Artiicial Intelligence (AAAI-19).

[4] Tom Zahavy, Nir Ben-Zrihem, and Shie Mannor. 2016. Graying the black box: Understanding DQNs. In Proceedings of the 33rd
International Conference on Machine Learning.


## References

This repo aim to provide quick access for paper and source. If you interested in XRL and RL, please read and cite this paper below. 

[1] Stephanie Milani, Nicholay Topin, Manuela Veloso, and Fei Fang. 2023. [Explainable Reinforcement Learning: A Survey and Comparative Review](https://doi.org/10.1145/3616864). ACM Comput. Surv. Just Accepted (August 2023). [https://doi.org/10.1145/3616864](https://doi.org/10.1145/3616864)

```
@article{10.1145/3616864,
author = {Milani, Stephanie and Topin, Nicholay and Veloso, Manuela and Fang, Fei},
title = {Explainable Reinforcement Learning: A Survey and Comparative Review},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {0360-0300},
url = {https://doi.org/10.1145/3616864},
doi = {10.1145/3616864},
abstract = {Explainable reinforcement learning (XRL) is an emerging subfield of explainable machine learning that has attracted considerable attention in recent years. The goal of XRL is to elucidate the decision-making process of reinforcement learning (RL) agents in sequential decision-making settings. Equipped with this information, practitioners can better understand important questions about RL agents (especially those deployed in the real world), such as what the agents will do and why. Despite increased interest, there exists a gap in the literature for organizing the plethora of papers — especially in a way that centers the sequential decision-making nature of the problem. In this survey, we propose a novel taxonomy for organizing the XRL literature that prioritizes the RL setting. We propose three high-level categories: feature importance, learning process and Markov decision process, and policy-level. We overview techniques according to this taxonomy, highlighting challenges and opportunities for future work. We conclude by using these gaps to motivate and outline a roadmap for future work.},
note = {Just Accepted},
journal = {ACM Comput. Surv.},
month = {aug},
keywords = {explainable reinforcement learning, interpretability, explainability}
}
```