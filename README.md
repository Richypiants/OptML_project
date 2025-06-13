# A Comparative Study of Preconditioner Update Strategies in the SOAP Optimizer

This project explores different precondition update strategies when training a deep neural network using the second-order optimizer SOAP.
The strategies were evaluated to determine if an alternative exists to using the standard constant update frequency.

Our experiments show that, in specific settings, choosing the right schedule for the update frequency can reduce computational costs while maintaining or improving training performance.

The project report included in this repository explains the experimental setup used to compare the different frequency schedules.

To generate the plots used in the report:
1. Run run_standard.py to generate plots for training on standard ResNet18;
2. Run run_dropoutSD.ipynb to generate plots for training the modified ResNet18 with dropout and stochastic depth added.
