Winning handover policy: `params_samo_1.pkl`. 

The policy was trained for 4.5M steps with learning rate $10^{-4}$, another 2M steps with learning rate $10^{-5}$, and then finetuned for 0.5M steps on an augmented reward function rewarding the myoHand for letting go of the object.