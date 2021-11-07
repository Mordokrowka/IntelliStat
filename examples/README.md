# Examples Documentation

## Table of Contents

 * [Examples](#examples)
 * [Examples structure](#example-structure)
 * [Implemented Examples](#implemented-examples)
   * [Function fitting](#function-fitting)
   * [Shape Learning](#shape-learning)
   * [Classifiers](#classifiers)

## Examples
- [__classifiers__](classifiers)
  - [__component\_classifier__](classifiers/component_classifier)
  - [__shape\_classifier__](classifiers/shape_classifier)
- [__function\_fitting__](function_fitting)
- [__shape\_learning__](shape_learning)

## Example Structure

- __example_name__
  - __resources__
    - config.json - config file for model builder - specified params of neural network and training process
    - config_schema.json - config.json validator - can be generated - https://www.liquid-technologies.com/online-json-to-schema-converter
  - example\_name.py - usage of IntelliStat code

## Implemented examples

### Function fitting

#### [Gauss fitting](function_fitting/gauss_fit.py)
Implementation of fitting to Gaussian function in 3 different ways - regression, method of moments and differential optimization.

#### [Linear fitting](function_fitting/linear_fit.py)
Implementation of fitting to Linear function in 2 different ways - linear regression and differential optimization.

### Shape Learning

Neural network model learns how shape looks like. Four shapes implemented:

- [Gauss](shape_learning/shape_learning_1G/shape_learning_1G.py)
- [Double Gauss](shape_learning/shape_learning_2G/shape_learning_2G.py)
- [Exp + Double Gauss](shape_learning/shape_learning_E2G/shape_learning_E2G.py)
- [Linear](shape_learning/shape_learning_linear/shape_learning_linear.py)

### Classifiers

#### [Component Classifier](classifiers/component_classifier/component_classifier.py)

Neural network model learns to detect what are the components of the shape.

#### [Shape Classifier](classifiers/shape_classifier/shape_classifier.py) 

Neural network model learns to discriminate between different shapes