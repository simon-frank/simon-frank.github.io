# Computer Scientist 


## Education		

**M.S. Computer Science @ Eberhard Karls University Tübingen (_October 2021_ - Present)**

**B.Eng. Computer Science (IT-Automotive) @ Baden-Wuerttemberg Cooperative State University (DHBW) Stuttgart (_October 2016_ - _September 2019_)**


## Work Experience

**Tool Developer @ Robert Bosch GmbH Abstatt (_November 2019_ - _October 2021_)** 
- Development and automation of toolchains for checking and determining metadata, integrating AI components such as computer vision to mark special situation such as construction sites, and storing data records from driver assistance systems (radar and video) for commercial vehicles
- Coordination of video and radar toolchain for determining and storing jointly recorded data records
- Adaptation of the toolchains to the different vehicles and countries of the endurance run
- Tool development for the automated adaptation of incorrect video, radar and label files to the requirements of the toolchains

**Working Student @ Robert Bosch GmbH Stuttgart (_October 2016_ - _September 2019_)**
- Development of software to operate a mobile spectrometer
- Implementation of a communication protocol between a machine diagnostic set and a Manufacturing
Execution System server for a log register with subsequent data mining
- Further development of a device network for the creation of a database of measured light spectra


## Projects 

### Encoding electronic health records with temporal transformers for clinical decision support
[Thesis](/assets/SimonFrank_MasterThesis.pdf)

In my master thesis I developed a novel approach to online multi-horizon clinical decision support using machine learning models trained solely on unstructured textual content and corresponding temporal data from electronic health recordsm (EHRs). The designed architecture employs hierarchical transformers with a late fusion strategy, leveraging a pre-trained LLM (Clinical Longformer) for note embeddings to reduce computational overhead. The temporal transformer enables the model to operate in an online mode and predict mortality in multiple rolling time horizons based on the textual content available in a patient’s EHR at any given point during their stay in the Intensive Care Unit (ICU). A multi-task learning objective was introduced to jointly predict mortality for all horizons, including short-term, medium-term, and total mortality. This objective allows the model to focus on predicting patient mortality across various time horizons. The model learns a shared representation of the patient across time horizons through the conditioning of long-term predictions on short-term predictions. This improves the performance of short-term predictions.

<p align="center">
  <img src="/assets/model_v2.JPG">
</p>

### Self-Supervised Learning for Medical Image Analysis

[Report](/assets/Self_Supervised_Learning.pdf)
[Presentation](/assets/Final%20Presentation_Abgabe.pdf)

In this group project we employ Barlow Twins pre-training on a ResNet-18 backbone as a self-supervised learning approach to generate transferable representations for medical image classification tasks. We attach a linear readout head to probe the feature vectors produced by the backbone and train the combined model in conjunction with two different fine-tuning strategies, namely, surgical and full fine-tuning. We evaluate the performance of our Barlow Twins model in comparison to a ResNet model pre-trained on ImageNet. Our experiment results demonstrate the superiority of our pre-trained features compared to the generic ImageNet based model on two different medical domains, showcasing the strength of utilizing Barlow Twins for enhancing performance in medical imaging classification tasks, particularly in the context of liver tumors and colorectal adenocarcinomas.

<p align="center">
  <img src="/assets/ssl.JPG">
</p>

### Sentiment analysis

[Report](/assets/NLP_Report.pdf)

Sentiment analysis plays a crucial role in understanding public opinion and sentiment expressed in textual data. In this project, the goal was to classify movie reviews as positive or negative using different models and feature sets. The dataset used was the Rotten Tomatoes dataset, which contains labelled movie reviews. The Naive Bayes models performed well, with an accuracy of about 76%. SVM models performed slightly worse than Naive Bayes, while LSTM models, especially with word2vec embeddings, showed improved accuracy, reaching around 77%. The BERT model, fine-tuned to the dataset,emerged as the best performing model with an accuracy of 80.30%.


### Counterfactual Explanations and the CARLA Library

[Report](/assets/Counterfactual_Explanations_and_the_CARLA_Library__Copy_.pdf)

There is a growing reliance on machine learning classifiers for important decisions like loan approvals, often seen as opaque black-box systems. Recourse methods are being developed to ensure human oversight and provide actionable steps to reverse adverse automated decisions, addressing the challenge of unpredictable changes and maintaining trust in decision-making processes. We implemented the Adversarially Robust Algorithmic Recourse (ARAR) algorithm in the Counterfactual And Recourse Library (CARLA). We were able to experimentally verify the robustness of ARAR, in particular in comparison to Wachter. We conducted experiments to further examine the effect of different hyperparameters to understand the inner workings of our ARAR implementation but also of the ARAR algorithm in general.

<p align="center">
  <img src="/assets/recourse.png">
</p>



### Cognitive Modeling

[Report](/assets/report_SimonFrank_research_project.pdf)

This research project was based on [Loci](https://arxiv.org/pdf/2205.13349.pdf), a self-supervised location and identity tracking system. Loci tackles the binding problem by processing separate, slot-wise encodings of ’what’, the Gestalt code, and ’where’, the position code. During the project, the Gestalt codes were analysed. It was shown that the binary Gestalt codes are disentangled, which may be a reason for the superior performance of Loci. Various regularisation techniques were also tested during the project, but none could outperform the binary version. The architecture of Loci was extended to include a prediction module in order to be able to runclosed loop with only the Gestalt and position codes. Through special training, the prediction was correct 20 time steps into the future using the CLEVRER dataset. This method provides a computationally fast and accurate way of reasoning at conceptual levels.

<p align="center">
  <img src="/assets/cl_run.JPG">
</p>

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

Cutibacterium acnes is a commensal bacterium playing a major role in healthcare associated infections, but also in a healthy skin microbiome. There are indications that C. acnes
also appears in the human nose environment. This project followed a pipeline to create
an genome-scale model for C. acnes, iPACH22FFG, to enable further in silico research.
In multiple steps the quality of the model was improved by increasing its consistency and
annotate it properly. In a specialized medium for the human nose (SNM3), the model
was able to grow. Sugar carbon sources increased its ability to grow. Thus, this project
resulted in an working GEM for C. acnes.

