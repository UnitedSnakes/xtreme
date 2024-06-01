from datasets import load_dataset

# dataset = load_dataset("tatoeba", lang1="en", lang2="pes", trust_remote_code=True)

# dataset = load_dataset("tatoeba", lang1="en", lang2="jv", trust_remote_code=True, date="v2020-11-09")["train"]
# print(dataset["train"]["translation"][0].keys())
# print(dataset["train"]["translation"][0].keys()[1])
# print(len(dataset["id"]))
# from datasets import Dataset

# # 假设你有一个Dataset对象叫做dataset
# data = {
#     'id': [1, 2, 3],
#     'ar': ['مرحبا', 'كيف حالك', 'مع السلامة'],
#     'en': ['hello', 'how are you', 'goodbye']
# }
# dataset = Dataset.from_dict(data)

# dataset = load_dataset("xtreme")
# dataset = load_dataset("xtreme", "XNLI")
dataset = load_dataset("multi_nli")
print(dataset)
# for i in dataset:
#     print(i)

# print(len("ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu".split(",")))