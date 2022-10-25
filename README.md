# Hands-On Machine Learning with Scikit-Learn and Scientific Python Toolkits

<a href="https://www.packtpub.com/data/hands-on-machine-learning-with-scikit-learn?utm_source=github&utm_medium=repository&utm_campaign=9781838826048"><img src="https://www.packtpub.com/media/catalog/product/cache/4cdce5a811acc0d2926d7f857dceb83b/9/7/9781838826048-original_44.png" alt="Hands-On Machine Learning with Scikit-Learn and Scientific Python Toolkits" height="256px" align="right"></a>

This is the code repository for [Hands-On Machine Learning with Scikit-Learn and Scientific Python Toolkits](https://www.packtpub.com/data/hands-on-machine-learning-with-scikit-learn?utm_source=github&utm_medium=repository&utm_campaign=9781838826048), published by Packt.

**A practical guide to implementing supervised and unsupervised machine learning algorithms in Python**

## What is this book about?
Machine learning is applied everywhere, from business to research and academia, while Scikit-Learn is a versatile library that is popular among machine learning practitioners. This book serves as a practical guide for anyone looking to provide hands-on machine learning solutions with Scikit-Learn and Python toolkits.

The book begins with an explanation of machine learning concepts and fundamentals, and strikes a balance between theoretical concepts and their applications. Each chapter covers a different set of algorithms, and shows you how to use them to solve real-life problems. You’ll also learn various key supervised and unsupervised machine learning algorithms using practical examples. Whether it is an instance-based learning algorithm, Bayesian estimation, a deep neural network, a tree-based ensemble, or a recommendation system, you’ll gain a thorough understanding of its theory and learn when to apply it. As you advance, you’ll learn how to deal with unlabeled data and when to use different clustering and anomaly detection algorithms.

By the end of this machine learning book, you’ll have learnt how to take a data-driven approach to provide end-to-end machine learning solutions. You’ll also have discovered how to formulate the problem at hand, prepare required data, and evaluate and deploy models in production.

This book goes beyond Scikit-Learn, and introduces you to complementary libraries such as NumPy, Pandas, SpaCy, imbalanced-learn, and Scikit-Surprise. The theoretical knowledge in this book should also prepare you to use libraries not mentioned here such as Tensor Flow and Pytorch.

In this repo, you will find the code examples used in the book. I also include here parts of the code omitted in the book, such as the data visualization styling, additional formatting, etc.

This book covers the following exciting features:
* Understand when to use supervised, unsupervised, or reinforcement learning algorithms
* Find out how to collect and prepare your data for machine learning tasks
* Tackle imbalanced data and optimize your algorithm for a bias or variance tradeoff
* Apply supervised and unsupervised algorithms to overcome various machine learning challenges
* Employ best practices for tuning your algorithm’s hyper parameters
* Discover how to use neural networks for classification and regression
* Build, evaluate, and deploy your machine learning solutions to production

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1838826041) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

## Book Reviews

### From [GoodReads](https://www.goodreads.com/book/show/54539914-hands-on-machine-learning-with-scikit-learn-and-scientific-python-toolki):

Ali Faizan rated it: 5 out of 5 stars.
> For a machine learning noob like me, it was pleasing to see that the book did not dive straight into the nitty-gritty of machine learning algorithms: it first established the raison d’être for machine learning and cohesively captured the whole gamut of developing a machine learning model. This helped me quite a bit to understand the bigger picture later on in the book where it demonstrated the practical use of various machine learning algorithms. I'll happily recommend this book to anyone interested in scikit-learn, and machine learning in general too

Paul Schmidt rated it: 5 out of 5 stars.
> This book is information rich with practical examples. I whom never read or touched this area was suprised to learn the weight that data analysis had on machine learning. Yes, this book also teaches you about data analysis. Throughout the chapters you learn what not to do when building machine learning and deep learning models. The author teaches you what not to do by analysing the data at hand and improving the models upon that knowledge. The book is very information rich and can easily be reread from chapter to chapter. There are some things to keep in mind, this book is not for python beginners and i urge you to know some of the basics from the pandas and matplotlib modules. In other words this book is strongly recommended. 

### From [Amazon](https://www.amazon.com/Machine-Learning-scikit-learn-Scientific-Toolkits-ebook/dp/B08BTFY8YW/):

Przemyslaw Chojecki rated it: 5 out of 5 stars.
> If you've already did a couple of data science projects, had a basic understanding of Python, did some visualisation and want to go deeper into some details of what it means to analyse data, then this book is for you. This is a practical guide to both supervised and unsupervised learning with plenty of examples in code. The main focus is on imperfect data and how to make sense of these imperfections through various machine learning algorithms. The author discusses standard data science algorithms using scikit-learn library which gives a coherent overview of the subjest. You will learn decision trees, KNN classification, Naive Bayes and much more; applied to classical datasets like Iris dataset, Boston housing prices or Fashion-MNIST. Recommended for beginning data scientists!


Adam Powell rated it: 5 out of 5 stars.
> The perfect read for an analyst that wants to transition into machine learning. It broadly covers all the key algorithms with an insightful practitioner's perspective. Highly recommended!

## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

```

**Following is what you need for this book:**
This book is for machine learning data scientists who want to master the theoretical and practical sides of machine learning algorithms and understand how to use them to solve real-life problems. Working knowledge of Python and a basic understanding of underlying mathematical and statistical concepts is required. Nevertheless, this book will walk you through the new concepts to cater to both new and experienced data scientists.

With the following software and hardware list you can run all code files present in the book (Chapter 1-13).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
| 1 - 13   |   Python 3.x, Jupyter Notebook/Google Colab                                         | Windows, Mac OS X, and Linux (Any) |


# Running the code

You will need Python 3.x installed on your computer. It is a good practice to set up a virtual environment to install the required libraries into. It's up to you whether you wish to use Python's venv module, the virtual environment provided by Anaconda, or any other option you like. We'll be using pip to install the libraries needed in the book, but once more, it is up to you whether you prefer to use conda or any other alternatives.

We suggest you create a conda environment first, then install the required libs there:

```
conda create -n scikitbook python=3.6
conda activate scikitbook
pip install --upgrade -r requirements.txt
```

You need to do the above steps once.
Then to activate the environment:

```
conda activate scikitbook
```

And to run Jupyter:

```
jupyter notebook
```

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781838826048_ColorImages.pdf).


### Related products <Other books you may enjoy>
* Python Machine Learning - Third Edition [[Packt]](https://www.packtpub.com/data/python-machine-learning-third-edition?utm_source=github&utm_medium=repository&utm_campaign=9781789955750) [[Amazon]](https://www.amazon.com/dp/1789955750)

* Mastering Machine Learning Algorithms - Second Edition [[Packt]](https://www.packtpub.com/data/mastering-machine-learning-algorithms-second-edition?utm_source=github&utm_medium=repository&utm_campaign=9781838820299) [[Amazon]](https://www.amazon.com/dp/B0843PMXPV)

## Get to Know the Author
**Tarek Amr**
has 8 years of experience in data science and machine learning. After finishing his postgraduate degree at the University of East Anglia, he worked in a number of startups and scaleup companies in Egypt and in the Netherlands. This is his second data-related book. His previous book is about data visualization using D3.js. He enjoys giving talks and writing about different computer science and business concepts and explaining them to a wider audience. He can be reached on twitter at [@gr33ndata](https://twitter.com/gr33ndata). He is happy to respond to all questions related to this book. Feel free to reach him if any parts of the book need clarifications or if you would like to discuss any of the concepts there in more detail.

You can also find the book's page on Good Reads [here](https://www.goodreads.com/book/show/54539914-hands-on-machine-learning-with-scikit-learn-and-scientific-python-toolki), your book reviews are highly appreciated.  

# Book Citation

Please make sure to cite the book if you use it in your research:

BiBTeX:

```
@book{amr2020hands,
  title={Hands-On Machine Learning with scikit-learn and Scientific Python Toolkits: A practical guide to implementing supervised and unsupervised machine learning algorithms in Python},
  author={Amr, Tarek},
  isbn={9781838823580},
  url={https://books.google.nl/books?id=GlbzDwAAQBAJ},
  year={2020},
  publisher={Packt Publishing, Limited}
}
```

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.
### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781838826048">https://packt.link/free-ebook/9781838826048 </a> </p>