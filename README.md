# Boosted-KDE

Boosted-KDE is a package for boosting the kernel density estimate (KDE) of numerical data. The notion of boosting the KDE has been proposed by Prof. [Marco Di Marzio](https://www.unich.it/ugov/person/1200) and Prof. [Charles Taylor](https://physicalsciences.leeds.ac.uk/staff/84/professor-charles-taylor). The aim of their original paper was to create a new classification algorithm based on KDE and boosting, named BoostKDC. Here, I implemented the algorithm outlined in Ref. [1] for KDE boosting to assign a weight to each observation in a dataset with the aim of detecting outliers or anomalies.

This algorithm rests on the idea of comparing the KDE computed with all samples with the KDE recomputed on all samples except the one of interest. In other words, the algorithm outputs a weight for each sample, which is equal to the log odds ratio between the full KDE and the correspoding leave-one-out estimate. 

Intuitevely, this algorithm has great potential in the context of outlier / anomaly detection, because there will be lower relative loss where observations are more frequent (the density is higher), whereas a major change in relative loss will occur where samples are more rare (the density is lower).

The "boosting" part of the algorithm was originally implemented as part of an AdaBoost classifier. However, in the context of outlier detection, which is unsupervised, it rarely shows benefits to run the algorithm more than once.


## Implementation

The python class `KDEBoosting` encapsulates the data on which to calculate the weights for each observation. It allows the user to:

+ Run boosting iterations not consecutively, which means that the algorithm can pick up the boosting process from the last iteration without having to restart from scratch
+ Plot outcome and report diagnostic information [*coming soon*]

KDE in highly dimensional feature spaces becomes quickly unfeasible with parametric methods. Therefore, this class computes the KDE with a non-parametric FFT method, as implemented in [KDEpy](https://kdepy.readthedocs.io/en/latest/index.html).


## Dependencies

The class `KDEBoosting ` depends on:

+ [KDEpy](https://kdepy.readthedocs.io/en/latest/index.html) for computing the KDE
+ [scikit-learn](https://scikit-learn.org/) for computing the KDE and for cross-validation
+ [numpy](https://www.numpy.org/) for fast array computations
+ [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) to generate graphs
+ [joblib](https://joblib.readthedocs.io/en/latest/) to process samples in parallel


# Usage

Simply import the class and pass your data to it:

    from boosted_KDE import KDEBoosting
    bKD = KDEBoosting(data)

Weights can be accessed via the attributes:

    .normalized_weights
    .weights    

The [theory notebook](notebooks/theory.ipynb) shows how the algorithm can give more or less weight to outliers.

The [tutorial notebook](notebooks/tutorial.ipynb) shows how the algorithm can be used on real world data in the context of (unsupervised) outlier / anomaly detection to improve the performance of an unsupervised one-class SVM classification model up to and over that of a (supervised) SVM classifier.


# License

Boosted-KDE is distributed under MIT license.


# References

[1] [Di Marzio and Taylor, *Biometrika* 2004](http://www1.maths.leeds.ac.uk/~charles/bka.pdf)


