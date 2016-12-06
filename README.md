# Artificial Intelligence Resources
Artificial Intelligence is advancing in an incredible fast pace and staying up to date with the state-of-the-art research is, sometimes, overwhelming.
This repository is my "reading list", a collection of interesting papers, courses, blogs, videos and other resources related to Machine Learning and Cognitive Systems in general.

## Table of Contents

<!-- MarkdownTOC depth=4 -->
- [Papers](#papers)
  - [Computational Cognitive Science](#cognitive-science)
  - [Computer Vision](#computer-vision)
  - [Deep Learning](#deep-learning)
  - [Hierarchical Temporal Memory](#htm)
  - [Self-Driving Cars](#sdc)
- [Websites & Blog posts](#blogs)
- [Videos](#videos)

<!-- /MarkdownTOC -->


<a name="papers" />
## Papers
<a name="cognitive-science" />
### Computational Cognitive Science
* [Building Machines That Learn and Think Like People](https://arxiv.org/abs/1604.00289) (2016) -  One one my favorite papers. Discusses the elements of human cognitition that are still missing in current Deep Learning and Reinforcement Learning approaches.

* [Human-level concept learning through probabilistic program induction](http://web.mit.edu/cocosci/Papers/Science-2015-Lake-1332-8.pdf) -  A different approach to learning concepts from few data, like humans do.

<a name="computer-vision" />
### Computer Vision
* [Learning What and Where to Draw](http://www.scottreed.info/files/nips2016.pdf) (2016) - "We propose a new model, the Generative Adversarial What-Where Network (GAWWN), that synthesizes images given instructions describing what content to draw in which location".
 
* [DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition](https://arxiv.org/abs/1310.1531) - "We evaluate whether features extracted from the activation of a deep convolutional network trained in a fully supervised fashion on a large, fixed set of object recognition tasks can be repurposed to novel generic tasks".

* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2015) - "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously".

* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (2015) - "Here we introduce an artificial system based on a Deep Neural Network that creates artistic images of high perceptual quality. The system uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images".

* [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-) (2012) - "We trained a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes".

<a name="deep-learning" />
### Deep Learning
* [Why does deep and cheap learning work so well?](https://arxiv.org/abs/1608.08225) (2016) - "HWe show how the success of deep learning depends not only on mathematics but also on physics: although well-known mathematical theorems guarantee that neural networks can approximate arbitrary functions well, the class of functions of practical interest can be approximated through "cheap learning" with exponentially fewer parameters than generic ones, because they have simplifying properties tracing back to the laws of physics".

* [One-shot Learning with Memory-Augmented Neural Networks](https://arxiv.org/abs/1605.06065) (2016) - "Here, we demonstrate the ability of a memory-augmented neural network to rapidly assimilate new data, and leverage this data to make accurate predictions after only a few samples. We also introduce a new method for accessing an external memory that focuses on memory content, unlike previous methods that additionally use memory location-based focusing mechanisms".

* [Learning to Compose Neural Networks for Question Answering](https://arxiv.org/abs/1601.01705) (2016) - "We describe a question answering model that applies to both images and structured knowledge bases. The model uses natural language strings to automatically assemble neural networks from a collection of composable modules. Our approach, which we term a dynamic neural model network, achieves state-of-the-art results on benchmark datasets in both visual and structured domains".

* [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) (2016) - "We introduce a guide to help deep learning practitioners understand and manipulate convolutional neural network architectures".

* [Action-Conditional Video Prediction using Deep Networks in Atari Games](https://sites.google.com/a/umich.edu/junhyuk-oh/action-conditional-video-prediction) (2015) - "Motivated by vision-based reinforcement learning (RL) problems, in particular Atari games from the recent benchmark Aracade Learning Environment (ALE), we consider spatio-temporal prediction problems where future (image-)frames are dependent on control variables or actions as well as previous frames".

* [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis) (2015) - "Progress in the field will be further accelerated by the development of better tools for visualizing and interpreting neural nets. We introduce two such tools here. The first is a tool that visualizes the activations produced on each layer of a trained convnet as it processes an image or video. (...) The second tool enables visualizing features at each layer of a DNN via regularized optimization in image space".

* [Wide & Deep Learning for Recommender Systems](http://arxiv.org/abs/1606.07792) - "In this paper, we present Wide & Deep learning - jointly trained wide linear models and deep neural networks - to combine the benefits of memorization and generalization for recommender systems".

<a name="htm" />
### Hierarchical Temporal Memory

* [Properties of Sparse Distributed Representations and their Application to Hierarchical Temporal Memory](https://arxiv.org/abs/1503.07469) - "Empirical evidence demonstrates that every region of the neocortex represents information using sparse activity patterns. This paper examines Sparse Distributed Representations (SDRs), the primary information representation strategy in Hierarchical Temporal Memory (HTM) systems and the neocortex".

* [HTM Whitepaper](http://numenta.com/assets/pdf/whitepapers/hierarchical-temporal-memory-cortical-learning-algorithm-0.2.1-en.pdf) - "Hierarchical Temporal Memory (HTM) is a technology modeled on how the neocortex performs these functions. HTM offers the promise of building machines that approach or exceed human-level performance for many cognitive tasks".

* [Biological and Machine Intelligence (BAMI)](http://numenta.com/biological-and-machine-intelligence/) - "Biological and Machine Intelligence (BAMI) is a living book authored by Numenta researchers and engineers. Its purpose is to document Hierarchical Temporal Memory, a theoretical framework for both biological and machine intelligence".

<a name="sdc" />
### Self-Driving Cars
* [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) (2016) - "We trained a convolutional neural network (CNN) to map raw pixels from a single front-facing camera directly to steering commands".

* [Learning a Driving Simulator](https://arxiv.org/abs/1608.01230) (2016) - "Comma.ai's approach to Artificial Intelligence for self-driving cars is based on an agent that learns to clone driver behaviors and plans maneuvers by simulating future events in the road. This paper illustrates one of our research approaches for driving simulation".

* [DeepDriving: Learning Affordance for Direct Perception in Autonomous Driving](http://deepdriving.cs.princeton.edu) (2015) - "Today, there are two major paradigms for vision-based autonomous driving systems: mediated perception approaches that parse an entire scene to make a driving decision, and behavior reflex approaches that directly map an input image to a driving action by a regressor. In this paper, we propose a third paradigm: a direct perception approach to estimate the affordance for driving. We propose to map an input image to a small number of key perception indicators that directly relate to the affordance of a road/traffic state for driving".

* [An Empirical Evaluation of Deep Learning on Highway Driving](https://arxiv.org/abs/1504.01716) (2015) - "In this paper, we presented a number of empirical evaluations of recent deep learning advances".

<a name="blogs" />
## Websites & Blog Posts

* [Montreal Institute for Learning Algorithms](https://mila.umontreal.ca/en/publications/) - MILA research publications.

* [Shakir's Machine Learning Blog](http://blog.shakirm.com) - Shakir is a senior research scientist at Google DeepMind in London.

* [A visual proof that neural nets can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html) - A very nice, simple and mostly visual explanation of the universality theorem, by Michael Nielsen.

* [Building a Deep Learning (Dream) Machine](http://graphific.github.io/posts/building-a-deep-learning-dream-machine/) - Guidelines for building a machine system specifically tailored for Deep Learning (2016), by Roelof Pieters.

### Computer Vision
* [Who is the best in dataset X?](http://rodrigob.github.io/are_we_there_yet/build/) - A collection of the best performing classification methods in datasets like MNIST and CIFAR-10.

<a name="videos" />
## Videos
* [Bay Area Deep Learning School - Day 1](https://www.youtube.com/watch?v=eyovmAtoUx0) (2016) - Day 1 of Bay Area Deep Learning School featuring speakers Hugo Larochelle, Andrej Karpathy, Richard Socher, Sherry Moore, Ruslan Salakhutdinov and Andrew Ng. Detailed schedule is at http://www.bayareadlschool.org/schedule.

* [Bay Area Deep Learning School - Day 2](https://www.youtube.com/watch?v=9dXiAecyJrY) (2016) - Day 2 of Bay Area Deep Learning School featuring speakers John Schulman, Pascal Lamblin, Adam Coates, Alex Wiltschko, Quoc Le and Yoshua Bengio. Detailed schedule is at http://www.bayareadlschool.org/schedule.
