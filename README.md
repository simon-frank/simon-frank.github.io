# Computer Scientist 


## Education								       		
- M.S., Computer Science | Eberhard Karls University Tübingen (_October 2021_ - Present)	 			        		
- B.Eng., Computer Science (IT-Automotive) | Baden-Wuerttemberg Cooperative State University (DHBW) Stuttgart (_October 2016_ - _September 2019_)


## Work Experience

** Tool Developer @ Robert Bosch GmbH ( _November 2019_ - _October 2021_)
- Development and automation of toolchains for checking and determining metadata and storing data records from driver assistance systems (radar and video) for commercial vehicles
- Coordination of video and radar toolchain for determining and storing jointly recorded data records
- Adaptation of the toolchains to the different vehicles and countries of the endurance run
- Adaptation of the toolchains to the different vehicles and countries of the endurance run.

** Working Student @ Robert Bosch GmbH (_October 2016_ - _September 2019_)
- Development of software to operate a mobile spectrometer
- Implementation of a communication protocol between a machine diagnostic set and a Manufacturing
Execution System server for a log register with subsequent data mining


## Projects {#projects}

During my master studies in Computer Science at the University of Tübingen I did a lot of projects. Some are listed below: 



### Research Project 

[Report](/assets/report_SimonFrank_research_project.pdf)

This research project was based on [Loci](https://arxiv.org/pdf/2205.13349.pdf), a self-supervised location and identity tracking system. Loci tackles the binding problem by processing separate, slot-wise encodings of ’what’, the Gestalt code, and ’where’, the position code. During the project, the Gestalt codes were analysed. It was shown that the binary Gestalt codes are disentangled, which may be a reason for the superior performance of Loci. Various regularisation techniques were also tested during the project, but none could outperform the binary version. The architecture of Loci was extended to include a prediction module in order to be able to runclosed loop with only the Gestalt and position codes. Through special training, the prediction was correct 20 time steps into the future using the CLEVRER dataset. This method provides a computationally fast and accurate way of reasoning at conceptual levels.

### Self-Supervised Learning for Medical Image Analysis

[Report](/assets/Self_Supervised_Learning.pdf)

In this group project we employ Barlow Twins pre-training on a ResNet-18 backbone as a self-supervised learning approach to generate transferable representations for medical image classification tasks. We attach a linear readout head to probe the feature vectors produced by the backbone and train the combined model in conjunction with two different fine-tuning strategies, namely, surgical and full fine-tuning. We evaluate the performance of our Barlow Twins model in comparison to a ResNet model pre-trained on ImageNet. Our experiment results demonstrate the superiority of our pre-trained features compared to the generic ImageNet based model on two different medical domains, showcasing the strength of utilizing Barlow Twins for enhancing performance in medical imaging classification tasks, particularly in the context of liver tumors and colorectal adenocarcinomas.

### Reinforcement Learning 

[Report](/assets/Report_ReinforcementRangers_SimonFrank.pdf)

This project used reinforcement learning to solve two environments. The first is a pendulum environment. The second is a hockey environment based on air hockey. There are two agents with a goal and a puck. The goal is to score a goal with the puck by moving it in the correct way. Both environments were solved using the Twin Delayed DDPG algorithm (TD3). The simple pendulum environment required a small actor and critic network. The hockey environment required more exploration noise and bigger actor and critic networks. I showed that TD3 has limitations. If the environment is complex, the maximum reward is limited by the networks.


### TüBing

[Report](/assets/search_engines_project.pdf)

In this group project, we created a search engine for content related to Tübingen. First, we crawled various pages. During the crawling process, we ensured that only Tübingen-related content was crawled. We used TF-IDF and BM25 as basic ranking methods. We used BERT to calculate the embedding for each page and query. We used the cosine similarity as a ranking score. To make our interface easier to read, we added a reading function that reads the title and abstract of search results when the user hovers over them.


### Does momentum affect an NBA team’s subsequent game result?

[Report](/assets/data_literacy.pdf)

In this report, we attempt to answer the question of whether or not the momentum an NBA team experiences, affects their next games result in a significant way. We, therefore, perform different chi-squared independence tests. The required match data is obtained in form of a dataset containing basketball results from the last 16 seasons. Every p-value of the performed chi-squared independence tests is significantly smaller than the chosen significance level of $5 \%$. Therefore every null hypothesis can be rejected, which implies, that for the given dataset it is very unlikely that there is no impact of momentum, no matter where a game takes place or which definition and streak length is applied. 


### System Biology I

[Report](/assets/project.pdf)
[Presentation](/assets/project_presentation.pdf)

Cutibacterium acnes is a commensal bacterium playing a major role in healthcare associated infections, but also in a healthy skin microbiome. There are indications that C. acnes
also appears in the human nose environment. This project followed a pipeline to create
an genome-scale model for C. acnes, iPACH22FFG, to enable further in silico research.
In multiple steps the quality of the model was improved by increasing its consistency and
annotate it properly. In a specialized medium for the human nose (SNM3), the model
was able to grow. Sugar carbon sources increased its ability to grow. Thus, this project
resulted in an working GEM for C. acnes.

