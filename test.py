# from bert_embedding import BertEmbedding
#
# word = "powerful"
# # sentences = bert_abstract.split('\n')
# bert_embedding = BertEmbedding()
# result = bert_embedding(word)
# print(type(result))
#
# from bert_embedding import BertEmbedding
# # ctx = mxnet.gpu()
# embedding = BertEmbedding()
# result = embedding(word)
# print(result)

# from bert_serving.client import BertClient
# bc = BertClient(ip='localhost')
# test=bc.encode(['bert','new'])
# print('===========')
# print(test[0])
# print('===========')
# test=bc.encode(['bert'])
# print(test[0])
# print('===========')
# test=bc.encode(['First do it', 'then do it right', 'then do it better'])
# print(test.shape)
# from service.client import BertClient
# bc = BertClient()
# bc.encode(['First do it', 'then do it right', 'then do it better'])

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# tokens = tokenizer.tokenize(text)
# print(tokens)
# encoded_input = tokenizer(text, return_tensors='pt')
# print(encoded_input)
# output = model(**encoded_input)
# print(output[0].shape)
#
# new_tokens = text.split()
# num_added_toks = tokenizer.add_tokens(new_tokens)
# model.resize_token_embeddings(len(tokenizer))
# tokenizer.save_pretrained("bert-base-uncased")
# print(tokenizer.tokenize('Replace'))
# # print(tokenizer.tokenize(text))
# encoded_input = tokenizer(text, return_tensors='pt')
# print(encoded_input)
# output = model(**encoded_input)
# print(output[0].shape)

# import torch
# from transformers import *
#
# class Bertvec:
#     def __init__(self, model_path, device, fix_embeddings=True):
#         self.device = device
#         self.model_class = BertModel
#         self.tokenizer_class = BertTokenizer
#         self.pretrained_weights = model_path
#         self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
#         self.model = self.model_class.from_pretrained(self.pretrained_weights).to(self.device)
#         if fix_embeddings:
#             for name, param in self.model.named_parameters():
#                 if name.startswith('embeddings'):
#                     param.requires_grad = False
#
#     def extract_features(self, input_batch_list):
#         batch_size = len(input_batch_list)
#         words = [sent for sent in input_batch_list]
#         word_seq_lengths = torch.LongTensor(list(map(len, words)))
#         # 每句句子的长度,获得最长长度
#         max_word_seq_len = word_seq_lengths.max().item()
#         word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
#         # 长度从长到短排列，并获得由原始排列到从长到短排列的转换顺序 eg:[2,3,1]句子长度，则转换顺序为[1,0,2]
#         batch_tokens = []
#         batch_token_ids = []
#         subword_word_indicator = torch.zeros((batch_size, max_word_seq_len), dtype=torch.int64)
#         for idx in range(batch_size):
#             one_sent_token = []
#             one_subword_word_indicator = []
#             for word in input_batch_list[idx]:
#                 word_tokens = self.tokenizer.tokenize(word)
#                 # 按照wordpiece分词
#                 one_subword_word_indicator.append(len(one_sent_token) + 1)
#                 # 由于分词之后，和输入的句子长度不同，因此需要解决这个问题，这里保存原始句子中词和分词之后的首个词的对应关系
#                 one_sent_token += word_tokens
#                 # 针对一句句子，获得分词后的结果
#             # 添加 [cls] and [sep] tokens
#             one_sent_token = ['[CLS]'] + one_sent_token + ['[SEP]']
#             one_sent_token_id = self.tokenizer.convert_tokens_to_ids(one_sent_token)
#             # token转换id
#             batch_tokens.append(one_sent_token)
#             batch_token_ids.append(one_sent_token_id)
#             subword_word_indicator[idx, :len(one_subword_word_indicator)] = torch.LongTensor(one_subword_word_indicator)
#         token_seq_lengths = torch.LongTensor(list(map(len, batch_tokens)))
#         max_token_seq_len = token_seq_lengths.max().item()
#         # 计算分词之后最长的句子长度
#         batch_token_ids_padded = []
#         for the_ids in batch_token_ids:
#             batch_token_ids_padded.append(the_ids + [0] * (max_token_seq_len - len(the_ids)))
#             # 补充pad
#         batch_token_ids_padded_tensor = torch.tensor(batch_token_ids_padded)[word_perm_idx].to(self.device)
#         subword_word_indicator = subword_word_indicator[word_perm_idx].to(self.device)
#         # 都按照之前得出的转换顺序改变为没有分词之前的句子从长到短的排列。
#         with torch.no_grad():
#             last_hidden_states = self.model(batch_token_ids_padded_tensor)[0]
#         # 提取bert词向量的输出
#         batch_word_mask_tensor_list = []
#         for idx in range(batch_size):
#             one_sentence_vector = torch.index_select(last_hidden_states[idx], 0, subword_word_indicator[idx]).unsqueeze(
#                 0)
#             # 根据对应关系，用分词之后的第一个分词来代表整个词，并添加batch的维度
#             batch_word_mask_tensor_list.append(one_sentence_vector)
#         batch_word_mask_tensor = torch.cat(batch_word_mask_tensor_list, 0)
#         return batch_word_mask_tensor
#
#     def save_model(self, path):
#         # 将网上下载的模型文件保存到path中
#         self.tokenizer.save_pretrained(path)
#         self.model.save_pretrained(path)
#
#
# if __name__ == '__main__':
#     input_test_list = [["he", "comes", "from", "--", "encode"],
#                        ["One", "way", "of", "measuring", "the", "complexity"],
#                        ["I", "encode", "money"]
#                        ]
#     bert_embedding = Bertvec('bert-base-uncased/', 'gpu', True)
#     batch_features = bert_embedding.extract_features(input_test_list)
#     print(batch_features)

from transformers import BertTokenizer, BertModel
import torch
DEVICE='cuda'
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(output)

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# model = BertModel.from_pretrained("bert-base-cased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(output)

# from transformers import BertTokenizer,AutoModel
#
# from transformers import AutoModel, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# model = AutoModel.from_pretrained('bert-base-cased')
# input_ids = tokenizer.encode('new', return_tensors='pt')
# last_hidden_state, _ = model(input_ids) # shape (1, 7, 768)
# print(model(input_ids))

# text = "After new that"
# marked_text = "[CLS] " + text + " [SEP]"
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tokenized_text = tokenizer.tokenize(marked_text)
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 得到每个词在词表中的索引
# segments_ids = [1] * len(tokenized_text)
# tokens_tensor = torch.tensor([indexed_tokens]).to(DEVICE)
# segments_tensors = torch.tensor([segments_ids]).to(DEVICE)
# model = BertModel.from_pretrained('bert-base-cased',
#                                   output_hidden_states=True)
# model.to(DEVICE)
# model.eval()
# with torch.no_grad():
#     outputs = model(tokens_tensor, segments_tensors)
#     hidden_states = outputs[2]
# 
# token_embeddings = torch.stack(hidden_states, dim=0)
# token_embeddings.size()
# token_embeddings = torch.squeeze(token_embeddings, dim=1)
# token_embeddings.size()
# token_embeddings = token_embeddings.permute(1, 0, 2)  # 调换顺序
# token_embeddings.size()

# 词向量表示
# token_vecs_cat = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
#                   token_embeddings]  # 连接最后四层 [number_of_tokens, 3072]
# token_vecs_sum = [torch.sum(layer[-4:], 0) for layer in token_embeddings]  # 对最后四层求和 [number_of_tokens, 768]
# print(token_vecs_cat[0].shape)

# sum_vec = torch.sum(token[-4:], dim=0)
# Use `sum_vec` to represent `token`.
# token_vecs_sum.append(sum_vec)
# print(len(token_vecs_sum))
# print(token_vecs_sum[0].shape)
# 句子向量表示
# token_vecs = hidden_states[-2][0]
# sentence_embedding = torch.mean(token_vecs, dim=0)  # 一个句子就是768维度
# print(sentence_embedding)

# f_clean = open('sts_test_word_sent6_bert_128.txt', "r", encoding='utf-8')
# line=f_clean.readlines()
# print(line)
# for i in line:
#     print(i)
import numpy as np

with open(r"sts_test_word_sent6_bert_128.txt", "r", encoding="utf-8") as f:
    for line in f:
        data = line.strip("\n").split('\t')
        data_list=data[1][1:len(data[1])-1].split(',')
        all_list=[]
        for i in range(len(data_list)):
            all_list.append(float(data_list[i]))
        all_numpy=np.array(all_list)
        
        print(type(all_numpy))
        exit()
        