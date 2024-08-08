import streamlit as st

st.set_page_config(page_title = "Data", layout = "wide")

st.title(":robot_face: Model Information")

st.markdown('''
            The model used for classifying between "Vaccinated" and "Not vaccinated" is a multi-layer Perceptron classifier which optinises the log-loss function using LBFGS or stochastic gradient descent.
            From [scikit-learn: 1.17.1. Multi-layer Perceptron](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) :link:

            \"
            Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function $f: \mathrm R^m \\to \mathrm R^o$ by training on a dataset, 
            where $m$ is the number of dimensions for input and $o$ is the number of dimensions for output. Given a set of features $X = x_1, x_2, \ldots, x_m$ and a target $y$, 
            it can learn a non-linear function approximator for either classification or regression. It is different from logistic regression, 
            in that between the input and the output layer, there can be one or more non-linear layers, called hidden layers. 

            The leftmost layer, known as the input layer, consists of a set of neurons $\{x_i | x_1, x_2, \ldots, x_m\}$
            representing the input features. Each neuron in the hidden layer transforms the values from the previous layer with a weighted linear summation
            $w_1 x_1 + w_2 x_2 + \ldots + w_m x_m $, 
            followed by a non-linear activation function $g(\cdot): \mathrm R \\to \mathrm R$ - like the hyperbolic tan function. 
            The output layer receives the values from the last hidden layer and transforms them into output values.
            \"
         ''')