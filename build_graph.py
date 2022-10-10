# sts_word_sent7: bert-768
# mr_word_sent6: bert-128
# mr_word_sent6: bert-768
# sst2_word_sent8: bert-768
# sst2_word_sent7: bert-768 random test
import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
import nltk,re
from nltk.corpus import wordnet
from treelib import Tree, Node
from stanfordcorenlp import StanfordCoreNLP
from parse import DisjointSet
from senticnet.senticnet import SenticNet
from bert_embedding import BertEmbedding
bert_embedding = BertEmbedding()
from bert_serving.client import BertClient
bc = BertClient(ip='localhost')
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
global bum
bum=1
nlp=StanfordCoreNLP(r'stanford-corenlp-4.4.0', lang='en')
# =========================
def get_dominateNode_and_maxDistance(sentence):
    tree = Tree()
    arcs = []
    words = []
    dominate_node = 0
    # with StanfordCoreNLP(r'stanford-corenlp-4.4.0', lang='en') as nlp:
    #     # print(nlp.parse(sentence))
    # try:
    #     parse_tree = nlp.parse(sentence)
    # except:
    #     nlp = StanfordCoreNLP(r'stanford-corenlp-4.4.0', lang='en')
    global bum, nlp
    if bum % 10000 == 1:
        nlp = StanfordCoreNLP(r'stanford-corenlp-4.4.0', lang='en')
        bum = 1
        print('bum=1')

    parse_tree = nlp.parse(sentence)
    parse_tree = parse_tree.strip()
    parse_tree = parse_tree.strip('\n')
    parse_tree = parse_tree.strip('\r')
    # print(parse_tree)
    dominate_node = parse_tree.count('(') - 2
    # print(nlp.dependency_parse(sentence))

    arcs = list(nlp.dependency_parse(sentence))
    words = list(nlp.word_tokenize(sentence))
    # Tree.fromstring(nlp.parse(sentence)).draw()

    arcs.sort(key=lambda x: x[2])
    # print(arcs)
    words = [w + "-" + str(idx) for idx, w in enumerate(words)]
    new_words = [idx + 1 for idx, w in enumerate(words)]
    rely_id = [arc[1] for arc in arcs]
    relation = [arc[0] for arc in arcs]
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]
    new_heads = [(id) for id in rely_id]
    dic = dict()
    n = len(words) + 1
    d = DisjointSet(int(n))
    final_list = []
    for i in range(len(words)):
        # print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')
        # print(relation[i] + '(' + str(new_words[i]) + ', ' + str(new_heads[i]) + ')')
        i, j = list(map(int, [new_words[i], new_heads[i]]))
        d.union(i, j)
        final_list = d.show()

    distance = 0
    for each in final_list:
        if each != 0:
            distance += 1
    return dominate_node, distance

def get_character_feature(word):
    # number_of_characters=len(word)
    # word_length可以代替
    if word.istitle()==True:
        start_with_capital_letter=1
    else:
        start_with_capital_letter=0

    have_alphanumeric_letters=0
    pattern = re.compile('[0-9]+')
    for v in word:
        match = pattern.findall(v)
        if match:
            have_alphanumeric_letters=1
            break
    if word.isupper()==True:
        capital_letters_only=1
    else:
        capital_letters_only =0
    return start_with_capital_letter,have_alphanumeric_letters,capital_letters_only

# =========================

if len(sys.argv) < 2:
    sys.exit("Use: python build_graph.py <dataset>")

# settings
datasets = ['easy_mr','mr','GR_Provo_mr','sentiment140','RR_split_rt','RR_split_Polarity','RR_split_mr','RR_split_SST2','split_SST1', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2']

dataset = sys.argv[1]
# if dataset not in datasets:
#     sys.exit("wrong dataset name")

try:
    window_size = int(sys.argv[2])
except:
    window_size = 3
    print('using default window size = 3')

try:
    weighted_graph = bool(sys.argv[3])
except:
    weighted_graph = False
    print('using default unweighted graph')

truncate = False  # whether to truncate long document
MAX_TRUNC_LEN = 350

print('loading raw data')

# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}
row=1
# with open('glove.840B/glove.840B.' + str(word_embeddings_dim) + 'd.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         print(row)
#         data = line.split(' ')
#         # vector=np.asarray(data[1:],'float32')
#         word_embeddings[str(data[0])] = list(map(float, data[1:]))
#         row+=1


# load document list
doc_name_list = []
doc_train_list = []
doc_test_list = []

with open('data/'+dataset+'/' + dataset + '.txt', 'r') as f:
    row=0
    for line in f.readlines():
        print(row)
        doc_name_list.append(line.strip())
        temp = line.split("\t")

        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
        row+=1

# load raw text
doc_content_list = []
print('clean.txt')
with open('data/'+ dataset +'/'+ dataset + '.clean.txt', 'rb') as f:
    row = 0
    for line in f.readlines():
        print(row)
        doc_content_list.append(line.decode('utf-8','ignore').strip())
        row+=1

# map and shuffle
print('train')
train_ids = []
row=0
for train_name in doc_train_list:
    # print(row)
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
    row+=1

random.shuffle(train_ids)
print('test')
row=0
test_ids = []
for test_name in doc_test_list:
    # print(row)
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
    row += 1
random.shuffle(test_ids)

ids = train_ids + test_ids

shuffle_doc_name_list = []
shuffle_doc_words_list = []

print('id')
row=0
for i in ids:
    # print(row)
    print(i)
    shuffle_doc_name_list.append(doc_name_list[int(i)])
    shuffle_doc_words_list.append(doc_content_list[int(i)])
    row += 1

# build corpus vocabulary
word_set = set()

print('shuffle')
row=0
for doc_words in shuffle_doc_words_list:
    # print(row)
    words = doc_words.split()
    word_set.update(words)
    row += 1

f_output=open('doc_list_2.txt','w')
for i in shuffle_doc_name_list:
    f_output.write(str(i)+'\n')

vocab = list(word_set)
vocab_size = len(vocab)
print('vocab')
row=0
word_id_map = {}
for i in range(vocab_size):
    # print(row)
    word_id_map[vocab[i]] = i
    row+=1

# initialize out-of-vocabulary word embeddings
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)

# build label list
label_set = set()
print('doc_meta')
row=0
for doc_meta in shuffle_doc_name_list:
    # print(row)
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
    row+=1
label_list = list(label_set)

# select 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_ids)


# build graph function
def build_graph(start, end):
    x_adj = []
    x_feature = []
    x_FFD_weight = []
    x_TRT_weight = []
    x_FFD1_weight = []
    x_TRT1_weight = []
    x_FFD2_weight = []
    x_TRT2_weight = []
    x_FFD3_weight = []
    x_TRT3_weight = []
    x_FFD4_weight = []
    x_TRT4_weight = []
    y = []
    doc_len_list = []
    vocab_set = set()

    '''
    [('number_of_characters', 14.319976937209452), ('is_entity_critical_word', 30.72829773070795),
     ('number_of_dominated_nodes', -0.2597880166249555), ('complexity_score', 0.07243095145876907),
     ('max_dependency_distance', 0.34931017626287786), ('number_of_senses_in_wordnet', 0.23235949626631536)]
     '''
    w_FFD = dict()
    w_TRT = dict()

    w_FFD['b'] = 48.0787070062293
    w_FFD['number_of_characters'] = 14.4777017083536
    w_FFD['start_with_capital_letter'] = -5.31835615569998
    w_FFD['is_entity_critical_word'] = 29.8971059550423
    w_FFD['number_of_senses_in_wordnet'] = 0.307357655863711

    w_TRT['b'] = 38.4023389234342
    w_TRT['number_of_characters'] = 27.4163637524332
    w_TRT['start_with_capital_letter'] = -9.27772659542626
    w_TRT['is_entity_critical_word'] = 34.598353758941
    w_TRT['number_of_senses_in_wordnet'] = 0.209969571306228
    w_TRT['number_of_dominated_nodes'] = -0.545413257253419
    w_TRT['complexity_score'] = 0.0791859296669166
    w_TRT['max_dependency_distance'] = 0.600986933398738
    # update_mr_3
    # w_FFD['b'] =48.12209683
    # w_FFD['number_of_characters'] = 14.4733710991442
    # w_FFD['start_with_capital_letter'] = -5.11205485072677
    # w_FFD['is_entity_critical_word'] = 29.5791575317618
    # w_FFD['number_of_senses_in_wordnet'] = 29.5791575317618
    # w_FFD['sentiment']=4.48408450751278
    #
    # w_TRT['b'] = 38.3808605784006
    # w_TRT['number_of_characters'] = 27.3109812956362
    # w_TRT['start_with_capital_letter'] = -9.32856094258717
    # w_TRT['is_entity_critical_word'] = 34.7937037880134
    # w_TRT['number_of_senses_in_wordnet'] = 0.169689111985213
    # w_TRT['sentiment']=12.5486335231536
    # w_TRT['number_of_dominated_nodes'] = -45.399041536875
    # w_TRT['complexity_score'] = 2.09826695415557
    # w_TRT['max_dependency_distance'] = 22.930050000977

    # # update_mr_2
    # w_FFD['b']=48.0787070062293
    # w_FFD['number_of_characters']=14.4777017083536
    # w_FFD['start_with_capital_letter']=-5.31835615569998
    # w_FFD['is_entity_critical_word'] =29.8971059550423
    # w_FFD['number_of_senses_in_wordnet'] =0.307357655863711
    #
    # w_TRT['b']=38.4023389234342
    # w_TRT['number_of_characters'] = 27.4163637524332
    # w_TRT['start_with_capital_letter'] =-9.27772659542626
    # w_TRT['is_entity_critical_word'] =34.598353758941
    # w_TRT['number_of_senses_in_wordnet'] =0.209969571306228
    # w_TRT['number_of_dominated_nodes'] =-0.545413257253419
    # w_TRT['complexity_score']=0.0791859296669166
    # w_TRT['max_dependency_distance'] =0.600986933398738
    # update_mr_1
    # w_FFD['b']=51.99896512
    # w_FFD['number_of_characters']=17.8513417692111
    # w_FFD['start_with_capital_letter']=-8.41838011068784
    # w_FFD['have_alphanumeric_letters'] =34.026754183144
    # w_FFD['capital_letters_only'] =-12.5436018964268
    #
    # w_TRT['b']=122.428485080425
    # w_TRT['is_entity_critical_word'] = 130.607304225225
    # w_TRT['number_of_dominated_nodes'] =-0.758952185868942
    # w_TRT['complexity_score'] =0.0586941130365055
    # w_TRT['max_dependency_distance'] =1.23163968304725
    # w_TRT['number_of_senses_in_wordnet'] =-2.2940762681708

    # update_mr
    # w_FFD['b']=49.7401712524012
    # w_FFD['number_of_characters']=14.0430319694729
    # w_FFD['is_entity_critical_word'] =33.3409606045504
    # w_FFD['start_with_capital_letter']=-5.19194387435582
    # w_FFD['have_alphanumeric_letters'] =53.085555562153
    # w_FFD['capital_letters_only'] =-17.6198851268182
    #
    # w_TRT['b']=35.2466859309982
    # w_TRT['number_of_dominated_nodes'] =-0.587999329480895
    # w_TRT['complexity_score'] =0.125210169336935
    # w_TRT['max_dependency_distance'] =0.632921896530374
    # w_TRT['number_of_senses_in_wordnet'] =1.08915833666803
    # RR
    # w_RR['b'] = 52.5320957495372
    # w_RR['number_of_characters'] = 14.3199769372094
    # w_RR['is_entity_critical_word'] = 30.72829773
    # w_RR['number_of_dominated_nodes'] = -0.259788017
    # w_RR['complexity_score'] = 0.072430951
    # w_RR['max_dependency_distance'] = 0.349310176
    # w_RR['number_of_senses_in_wordnet'] = 0.232359496

    # GR
    # w_RR['b'] = 135.38312218
    # w_RR['number_of_characters'] = 35.7092717950976
    # w_RR['is_entity_critical_word'] = 14.7568429853783
    # w_RR['number_of_dominated_nodes'] = -3.11089762291366
    # w_RR['complexity_score'] = -0.978330501004471
    # w_RR['max_dependency_distance'] = 3.44704523370297
    # w_RR['number_of_senses_in_wordnet'] = 1.9634755757931

    for i in tqdm(range(start, end)):
        global bum
        bum += 1
        doc_words = shuffle_doc_words_list[i].split()
        number_of_dominated_nodes, max_dependency_distance = get_dominateNode_and_maxDistance(shuffle_doc_words_list[i])
        complex_score = len(doc_words)

        function_list = ['CC', 'IN', 'LS', 'TO', 'POS', 'RP', 'SYM', 'UH']
        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        # 对每个word进行处理，计算word eye feature
        FFD_feature_list = []
        TRT_feature_list = []

        FFD1_feature_list = []
        TRT1_feature_list = []

        FFD2_feature_list = []
        TRT2_feature_list = []

        FFD3_feature_list = []
        TRT3_feature_list = []

        FFD4_feature_list = []
        TRT4_feature_list = []
        gaze_total = 0
        FFD_weight_map = dict()
        TRT_weight_map = dict()

        FFD1_weight_map = dict()
        TRT1_weight_map = dict()

        FFD2_weight_map = dict()
        TRT2_weight_map = dict()

        FFD3_weight_map = dict()
        TRT3_weight_map = dict()

        FFD4_weight_map = dict()
        TRT4_weight_map = dict()

        id_word_map={}

        for j in range(len(doc_words)):
            id_word_map[j]=doc_words[j]
            number_of_characters = len(doc_words[j])
            is_entity_critical_word = 1
            each_word_list = list(doc_words[j])
            pos_tags = nltk.pos_tag(each_word_list)[0][1]
            if pos_tags in function_list:
                is_entity_critical_word = 0
            number_of_sense_in_wordnet = len(wordnet.synsets(doc_words[j]))
            start_with_capital_letter, have_alphanumeric_letters, capital_letters_only = get_character_feature(doc_words[j])
            sn = SenticNet()
            try:
                sentiment = sn.polarity_value(doc_words[j])
            except:
                sentiment = 0
            try:
                sentiment1 = sn.polarity_value(doc_words[j])+1
            except:
                sentiment1 = 0+1
            # print(type(w_FFD['sentiment']) , type(sentiment))
            # FFD_feature = w_FFD['b'] + \
            #                     w_FFD['number_of_characters'] * number_of_characters + \
            #                     w_FFD['start_with_capital_letter'] * start_with_capital_letter+ \
            #                     w_FFD['is_entity_critical_word'] * is_entity_critical_word+ \
            #                     w_FFD['number_of_senses_in_wordnet'] * number_of_sense_in_wordnet+\
            #                     w_FFD['sentiment']* float(sentiment)
            #
            # TRT_feature = w_TRT['b']+ \
            #                     w_TRT['number_of_characters'] * number_of_characters + \
            #                     w_TRT['start_with_capital_letter'] * start_with_capital_letter + \
            #                     w_TRT['is_entity_critical_word'] * is_entity_critical_word + \
            #                     w_TRT['number_of_senses_in_wordnet'] * number_of_sense_in_wordnet+\
            #                     w_TRT['number_of_dominated_nodes'] * number_of_dominated_nodes + \
            #                     w_TRT['complexity_score'] * complex_score + \
            #                     w_TRT['max_dependency_distance'] * max_dependency_distance+ \
            #                     w_TRT['sentiment'] * float(sentiment)
            FFD_feature = w_FFD['b'] + \
                          w_FFD['number_of_characters'] * (number_of_characters + float(sentiment)) + \
                          w_FFD['start_with_capital_letter'] * (start_with_capital_letter + float(sentiment)) + \
                          w_FFD['is_entity_critical_word'] * (is_entity_critical_word + float(sentiment)) + \
                          w_FFD['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet + float(sentiment))  # +\
            # float(sentiment)
            # w_FFD['sentiment']* float(sentiment)

            TRT_feature = w_TRT['b'] + \
                          w_TRT['number_of_characters'] * (number_of_characters + float(sentiment)) + \
                          w_TRT['start_with_capital_letter'] * (start_with_capital_letter + float(sentiment)) + \
                          w_TRT['is_entity_critical_word'] * (is_entity_critical_word + float(sentiment)) + \
                          w_TRT['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet + float(sentiment)) + \
                          w_TRT['number_of_dominated_nodes'] * (number_of_dominated_nodes + float(sentiment)) + \
                          w_TRT['complexity_score'] * (complex_score + float(sentiment)) + \
                          w_TRT['max_dependency_distance'] * (max_dependency_distance + float(sentiment))
            # float(sentiment)
            # w_TRT['sentiment'] * float(sentiment)
            FFD3_feature = w_FFD['b'] + \
                          w_FFD['number_of_characters'] * (number_of_characters + float(sentiment1)) + \
                          w_FFD['start_with_capital_letter'] * (start_with_capital_letter + float(sentiment1)) + \
                          w_FFD['is_entity_critical_word'] * (is_entity_critical_word + float(sentiment1)) + \
                          w_FFD['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet + float(sentiment1))  # +\
            # float(sentiment)
            # w_FFD['sentiment']* float(sentiment)

            TRT3_feature = w_TRT['b'] + \
                          w_TRT['number_of_characters'] * (number_of_characters + float(sentiment1)) + \
                          w_TRT['start_with_capital_letter'] * (start_with_capital_letter + float(sentiment1)) + \
                          w_TRT['is_entity_critical_word'] * (is_entity_critical_word + float(sentiment1)) + \
                          w_TRT['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet + float(sentiment1)) + \
                          w_TRT['number_of_dominated_nodes'] * (number_of_dominated_nodes + float(sentiment1)) + \
                          w_TRT['complexity_score'] * (complex_score + float(sentiment1)) + \
                          w_TRT['max_dependency_distance'] * (max_dependency_distance + float(sentiment1))


            FFD1_feature = w_FFD['b'] + \
                           w_FFD['number_of_characters'] * (number_of_characters) + \
                           w_FFD['start_with_capital_letter'] * (start_with_capital_letter) + \
                           w_FFD['is_entity_critical_word'] * (is_entity_critical_word) + \
                           w_FFD['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet)  # +\
            # float(sentiment)
            # w_FFD['sentiment']* float(sentiment)

            TRT1_feature = w_TRT['b'] + \
                           w_TRT['number_of_characters'] * (number_of_characters) + \
                           w_TRT['start_with_capital_letter'] * (start_with_capital_letter) + \
                           w_TRT['is_entity_critical_word'] * (is_entity_critical_word) + \
                           w_TRT['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet) + \
                           w_TRT['number_of_dominated_nodes'] * (number_of_dominated_nodes) + \
                           w_TRT['complexity_score'] * (complex_score) + \
                           w_TRT['max_dependency_distance'] * (max_dependency_distance)
            # float(sentiment)
            # w_TRT['sentiment'] * float(sentiment)

            FFD2_feature = w_FFD['b'] + \
                           w_FFD['number_of_characters'] * (number_of_characters) + \
                           w_FFD['start_with_capital_letter'] * (start_with_capital_letter) + \
                           w_FFD['is_entity_critical_word'] * (is_entity_critical_word) + \
                           w_FFD['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet) + \
                           float(sentiment)
            # w_FFD['sentiment']* float(sentiment)

            TRT2_feature = w_TRT['b'] + \
                           w_TRT['number_of_characters'] * (number_of_characters) + \
                           w_TRT['start_with_capital_letter'] * (start_with_capital_letter) + \
                           w_TRT['is_entity_critical_word'] * (is_entity_critical_word) + \
                           w_TRT['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet) + \
                           w_TRT['number_of_dominated_nodes'] * (number_of_dominated_nodes) + \
                           w_TRT['complexity_score'] * (complex_score) + \
                           w_TRT['max_dependency_distance'] * (max_dependency_distance) + \
                           float(sentiment)

            FFD4_feature = w_FFD['b'] + \
                           w_FFD['number_of_characters'] * (number_of_characters) + \
                           w_FFD['start_with_capital_letter'] * (start_with_capital_letter) + \
                           w_FFD['is_entity_critical_word'] * (is_entity_critical_word) + \
                           w_FFD['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet) + \
                           float(sentiment1)
            # w_FFD['sentiment']* float(sentiment)

            TRT4_feature = w_TRT['b'] + \
                           w_TRT['number_of_characters'] * (number_of_characters) + \
                           w_TRT['start_with_capital_letter'] * (start_with_capital_letter) + \
                           w_TRT['is_entity_critical_word'] * (is_entity_critical_word) + \
                           w_TRT['number_of_senses_in_wordnet'] * (number_of_sense_in_wordnet) + \
                           w_TRT['number_of_dominated_nodes'] * (number_of_dominated_nodes) + \
                           w_TRT['complexity_score'] * (complex_score) + \
                           w_TRT['max_dependency_distance'] * (max_dependency_distance) + \
                           float(sentiment1)

            FFD_weight_map[j] = FFD_feature
            TRT_weight_map[j] = TRT_feature
            FFD1_weight_map[j] = FFD1_feature
            TRT1_weight_map[j] = TRT1_feature
            FFD2_weight_map[j] = FFD2_feature
            TRT2_weight_map[j] = TRT2_feature
            FFD3_weight_map[j] = FFD3_feature
            TRT3_weight_map[j] = TRT3_feature
            FFD4_weight_map[j] = FFD4_feature
            TRT4_weight_map[j] = TRT4_feature
            # word_gaze_feature_list.append(word_gaze_feature)

        # for j in range(len(word_gaze_feature_list)):
        #     word_gaze_feature_list[j]/=gaze_total

        doc_len = len(doc_words)  # sentence length

        # set会去掉重复出现的word，这时候可以采用一下映射
        doc_vocab = doc_words
        features = bc.encode(doc_vocab)
        # doc_vocab = list(set(doc_words))
        for j in range(len(doc_vocab)):
            FFD_feature_list.append(FFD_weight_map[j])
            TRT_feature_list.append(TRT_weight_map[j])
            FFD1_feature_list.append(FFD1_weight_map[j])
            TRT1_feature_list.append(TRT1_weight_map[j])
            FFD2_feature_list.append(FFD2_weight_map[j])
            TRT2_feature_list.append(TRT2_weight_map[j])
            FFD3_feature_list.append(FFD3_weight_map[j])
            TRT3_feature_list.append(TRT3_weight_map[j])
            FFD4_feature_list.append(FFD4_weight_map[j])
            TRT4_feature_list.append(TRT4_weight_map[j])
        # doc_vocab = doc_words
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        window_begin_id={}
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
            window_begin_id[str(doc_words)]=0
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                window_begin_id[str(window)]=j
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            begin_id=window_begin_id[str(window)]
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_q = window[q]
                    for k in range(begin_id,len(window)):
                        # print(doc_vocab[k],word_p)
                        if doc_vocab[k]==word_p:
                            word_p_id=k
                        if doc_vocab[k] == word_q:
                            word_q_id = k
                    # word_p_id = word_id_map[word_p]
                    # word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        # features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(p)
            col.append(q)
            # row.append(doc_word_id_map[vocab[p]])
            # col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        # print(len(row),len(col),doc_nodes)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))

        # for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            # print('v',v)
            # features.append(word_embeddings[k] if k in word_embeddings else oov[k])
            # result = bert_embedding(k)
            # bc = BertClient(ip='localhost')
            # result = bc.encode([k])
            # print(result)
            # features.append(result[0])
        x_adj.append(adj)
        x_feature.append(features)
        x_FFD_weight.append(FFD_feature_list)
        x_TRT_weight.append(TRT_feature_list)
        x_FFD1_weight.append(FFD1_feature_list)
        x_TRT1_weight.append(TRT1_feature_list)
        x_FFD2_weight.append(FFD2_feature_list)
        x_TRT2_weight.append(TRT2_feature_list)
        x_FFD3_weight.append(FFD3_feature_list)
        x_TRT3_weight.append(TRT3_feature_list)
        x_FFD4_weight.append(FFD4_feature_list)
        x_TRT4_weight.append(TRT4_feature_list)
        # print(x_FFD_weight)
    # one-hot labels
    for i in range(start, end):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)

    return x_adj, x_feature, y, doc_len_list, vocab_set, x_FFD_weight,x_TRT_weight,x_FFD1_weight,x_TRT1_weight,x_FFD2_weight,x_TRT2_weight,x_FFD3_weight,x_TRT3_weight,x_FFD4_weight,x_TRT4_weight

print('building graphs for training')
x_adj, x_feature, y, _, _, x_FFD_w,x_TRT_w,x_FFD1_w,x_TRT1_w,x_FFD2_w,x_TRT2_w,x_FFD3_w,x_TRT3_w,x_FFD4_w,x_TRT4_w = build_graph(start=0, end=real_train_size)
with open("data/{}/ind.{}.x_adj".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_adj, f)
with open("data/{}/ind.{}.x_embed".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_feature, f)
with open("data/{}/ind.{}.x_FFD_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_FFD_w, f)
with open("data/{}/ind.{}.x_TRT_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_TRT_w, f)
with open("data/{}/ind.{}.x_FFD1_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_FFD1_w, f)
with open("data/{}/ind.{}.x_TRT1_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_TRT1_w, f)
with open("data/{}/ind.{}.x_FFD2_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_FFD2_w, f)
with open("data/{}/ind.{}.x_TRT2_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_TRT2_w, f)
with open("data/{}/ind.{}.x_FFD3_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_FFD3_w, f)
with open("data/{}/ind.{}.x_TRT3_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_TRT3_w, f)
with open("data/{}/ind.{}.x_FFD4_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_FFD4_w, f)
with open("data/{}/ind.{}.x_TRT4_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_TRT4_w, f)
with open("data/{}/ind.{}.y".format(dataset, dataset), 'wb') as f:
    pkl.dump(y, f)

print('building graphs for training + validation')
allx_adj, allx_feature, ally, doc_len_list_train, vocab_train, allx_FFD_w, allx_TRT_w, allx_FFD1_w, allx_TRT1_w, allx_FFD2_w, allx_TRT2_w, allx_FFD3_w, allx_TRT3_w, allx_FFD4_w, allx_TRT4_w = build_graph(start=0, end=train_size)
with open("data/{}/ind.{}.allx_adj".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_adj, f)
with open("data/{}/ind.{}.allx_embed".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_feature, f)
with open("data/{}/ind.{}.allx_FFD_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_FFD_w, f)
with open("data/{}/ind.{}.allx_TRT_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_TRT_w, f)
with open("data/{}/ind.{}.allx_FFD1_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_FFD1_w, f)
with open("data/{}/ind.{}.allx_TRT1_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_TRT1_w, f)
with open("data/{}/ind.{}.allx_FFD2_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_FFD2_w, f)
with open("data/{}/ind.{}.allx_TRT2_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_TRT2_w, f)
with open("data/{}/ind.{}.allx_FFD3_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_FFD3_w, f)
with open("data/{}/ind.{}.allx_TRT3_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_TRT3_w, f)
with open("data/{}/ind.{}.allx_FFD4_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_FFD4_w, f)
with open("data/{}/ind.{}.allx_TRT4_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_TRT4_w, f)
with open("data/{}/ind.{}.ally".format(dataset, dataset), 'wb') as f:
    pkl.dump(ally, f)

print('building graphs for test')
tx_adj, tx_feature, ty, doc_len_list_test, vocab_test, tx_FFD_w, tx_TRT_w, tx_FFD1_w, tx_TRT1_w , tx_FFD2_w, tx_TRT2_w, tx_FFD3_w, tx_TRT3_w, tx_FFD4_w, tx_TRT4_w= build_graph(start=train_size, end=train_size + test_size)
with open("data/{}/ind.{}.tx_adj".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_adj, f)
with open("data/{}/ind.{}.tx_embed".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_feature, f)
with open("data/{}/ind.{}.tx_FFD_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_FFD_w, f)
with open("data/{}/ind.{}.tx_TRT_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_TRT_w, f)
with open("data/{}/ind.{}.tx_FFD1_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_FFD1_w, f)
with open("data/{}/ind.{}.tx_TRT1_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_TRT1_w, f)
with open("data/{}/ind.{}.tx_FFD2_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_FFD2_w, f)
with open("data/{}/ind.{}.tx_TRT2_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_TRT2_w, f)
with open("data/{}/ind.{}.tx_FFD3_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_FFD3_w, f)
with open("data/{}/ind.{}.tx_TRT3_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_TRT3_w, f)
with open("data/{}/ind.{}.tx_FFD4_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_FFD4_w, f)
with open("data/{}/ind.{}.tx_TRT4_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_TRT4_w, f)
with open("data/{}/ind.{}.ty".format(dataset, dataset), 'wb') as f:
    pkl.dump(ty, f)

'''
print('building graphs for training')
x_adj, x_feature, y, _, _, x_FFD_w,x_TRT_w = build_graph(start=0, end=real_train_size)
# dump objects
with open("data/{}/ind.{}.x_adj".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_adj, f)

with open("data/{}/ind.{}.x_embed".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_feature, f)

with open("data/{}/ind.{}.x_FFD_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_FFD_w, f)

with open("data/{}/ind.{}.x_TRT_weight".format(dataset,dataset), 'wb') as f:
    pkl.dump(x_TRT_w, f)

with open("data/{}/ind.{}.y".format(dataset, dataset), 'wb') as f:
    pkl.dump(y, f)


print('building graphs for training + validation')
allx_adj, allx_feature, ally, doc_len_list_train, vocab_train, allx_FFD_w, allx_TRT_w = build_graph(start=0, end=train_size)
with open("data/{}/ind.{}.allx_adj".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_adj, f)

with open("data/{}/ind.{}.allx_embed".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_feature, f)

with open("data/{}/ind.{}.allx_FFD_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_FFD_w, f)

with open("data/{}/ind.{}.allx_TRT_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(allx_TRT_w, f)

with open("data/{}/ind.{}.ally".format(dataset, dataset), 'wb') as f:
    pkl.dump(ally, f)

print('building graphs for test')
tx_adj, tx_feature, ty, doc_len_list_test, vocab_test, tx_FFD_w, tx_TRT_w = build_graph(start=train_size, end=train_size + test_size)
doc_len_list = doc_len_list_train + doc_len_list_test

with open("data/{}/ind.{}.tx_adj".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_adj, f)

with open("data/{}/ind.{}.tx_embed".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_feature, f)

with open("data/{}/ind.{}.tx_FFD_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_FFD_w, f)

with open("data/{}/ind.{}.tx_TRT_weight".format(dataset, dataset), 'wb') as f:
    pkl.dump(tx_TRT_w, f)

with open("data/{}/ind.{}.ty".format(dataset, dataset), 'wb') as f:
    pkl.dump(ty, f)


# statistics
print('max_doc_length', max(doc_len_list), 'min_doc_length', min(doc_len_list),
      'average {:.2f}'.format(np.mean(doc_len_list)))
print('training_vocab', len(vocab_train), 'test_vocab', len(vocab_test),
      'intersection', len(vocab_train & vocab_test))

'''