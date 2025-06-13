# A Comparative Study of Preconditioner Update Strategies in the SOAP Optimizer

In this project different precondition update strategies when training a deep neural network using the second-order optimizer SOAP.
THe strategies where evaluated to determine if an alternative exists to using the now standard constant update frequency.

Our experiments show that, in specific settings,choosing the right schedule for the update frequency can reducecomputational costs while maintaining or improving trainingperformance.

This project was summarized in the project report attached in this project.

To generate the plots used in the report:
1. Run run_standard.py to generate plots for training on standard ResNet18
2. Run run_dropoutSD.jupyternb to generate plots for training the modifed ResNet18 with stochastic depth and dropout.
