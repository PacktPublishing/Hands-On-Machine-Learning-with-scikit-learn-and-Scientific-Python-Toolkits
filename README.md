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
  author={Amr, T.},
  isbn={9781838823580},
  url={https://books.google.nl/books?id=GlbzDwAAQBAJ},
  year={2020},
  publisher={Packt Publishing}
}
```

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.
