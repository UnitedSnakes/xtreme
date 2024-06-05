# from datasets import load_dataset

# # dataset = load_dataset("xtreme", "XNLI")
# dataset = load_dataset("multi_nli")["validation_matched"]
# for ex in dataset:
#     assert ex["label"] in [0, 1, 2]

# print(len("ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu".split(",")))

# import numpy as np

# logits = np.array([[-0.09655189, -0.28757542,  0.088696  ],
#  [-0.08719115, -0.08453214,  0.04470615],
#  [-0.08110631, -0.1941469,  -0.00338058],
#  [-0.08117431, -0.13639034, -0.05045442],
#  [-0.08574901, -0.16250753, -0.06348856]])

# print(logits)
# print(type(logits))

# predictions = np.argmax(logits, axis=-1)
# print(predictions)
# import os


# a = 'fine_tuned_models/MBERT'
# b = os.path.isdir(os.path.join(a, '20240602_022155'))
# print(b)
import torch as th
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

print(DEVICE)