from math import radians
import numpy as np
import matplotlib.pyplot as plt

losses = [[15, 0],[5, 1], [1, 40],[0, 150]]
actions_names = ['Important','Show', 'Folder', 'Delete']
num_actions = len(losses)
prob_range = np.linspace(0, 1, num=600) 

def expected_loss_of_action(prob_spam, action):
    losses_given_action = losses[action] # loss lunction of a particular action
    expected_loss = (1 - prob_spam)*losses_given_action[1]+prob_spam*losses_given_action[0]
    return expected_loss

# Plotting starts here
for action in range(num_actions):
    plt.plot(prob_range, expected_loss_of_action(prob_range, action), label=actions_names[action])

plt.xlabel('$p(spam|email)$')
plt.ylabel('Expected loss of action')
plt.legend()
plt.title('The expected wasted user time for each of the four possible actions')
plt.show()
