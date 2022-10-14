[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Js1w67q_ZM9ODL7iRKM4AobODfJzZQbt)

# Overview

* In this Repo's Notebook I have implemented `SAM optimizer` algorithm from the research paper titled as [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) using tensorflow framework.

* The notebook explains working of SAM algorithm and steps involved in it.

* The code in this notebook is reusable and can be reused with any model or dataset of your choice.

* In this notebook I will use SAM with SGD for training ResNet50 model on Cifar10 dataset.

# <span style="color:coral">The problem of generalization:</span>

* The loss surface of deep networks tends to have many local minima. Many of them may be equally good in terms of training errors, but may have very different generalization performance.

* Generally neural network optimizers try to seek parameters that just minimize the loss but this does not gurantee us about good generalization results on test data. 

* To solve this problem an effective algorithm SAM is suggested in research paper [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412).

* Sharpness Aware Minimization abbreviated as SAM is an optimizer algorithm which seeks parameters that not only minimizes the loss but also seeks parameters that converge to a flatter minima thus increasing the generalization on test set.

# <center> <span style="color:coral">Algorithm and working</span> </center>

The SAM algorithm can be divided into two steps:

1. Calculate adversial point $\large{w}_{adv}$ *having highest* loss in neighbourhood $\rho$ by computing $\large\hat{\epsilon}$, i.e in this step we temporarily move from intial paramter $\large w_{t}$ to the adversial point $\large w_{adv}$ and compute the gradient at $\large w_{adv}$.

> **Note** if we were using normal gradient descent we would have updated the parameter $\large w_{t}$  on basis gradient at $\large w_{t}$ and learning rate $\eta$ and after the update we would go from $\large w_{t}$ to $\large w_{t+1}$ as compared to two steps required in SAM. 
    
2. From the gradient computed at adversial point $\large w_{adv}$ in previous step we update the intial parameter $\large w_{t}$ by learning rate $\eta$ in direction of  gradient at $\large w_{adv}$ so after the update we reach at point $\large w_{t+1}^{SAM}$ thus performing Sharpness Aware update.

# <center> <span style="color:coral">Implementation</span> </center>

* For implementing `SAM optimizer` I will use Model [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) to `override` ***the train_step*** method of model with custom ***train_step_sam*** so this can be implemented when we call fit on our model.

> **Note:** To learn more about customizing train_step try reading this [keras documentation](https://keras.io/guides/customizing_what_happens_in_fit/#a-first-simple-example)

* SAM has only one hyperparameter $\large\rho$, as mentioned earlier it defines the neighborhood size, where we do min max operation.

* Authors have found that $\large\rho$ = 0.05 gives good result on wide range of datasets, for current implementation we will use the same value but you can tune it as per your requirements.

* I will be using [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function) model by adding a Global Average pooling layer and a new output layer. 

```bibtex
@ARTICLE{2020arXiv201001412F,
       author = {{Foret}, Pierre and {Kleiner}, Ariel and {Mobahi}, Hossein and {Neyshabur}, Behnam},
        title = "{Sharpness-Aware Minimization for Efficiently Improving Generalization}",
         year = 2020,
          eid = {arXiv:2010.01412},
       eprint = {2010.01412},
}
```
