import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import stanford_parser
from sklearn.preprocessing import normalize
stanford_obj = stanford_parser.StanfordNLP()

def get_parser_output(sent):
    parse_output = stanford_obj.annotate(sent)
    sentences = parse_output['sentences'][0]
    basic_dependent = (sentences['basicDependencies'])
    tokens = sentences['tokens']
    return basic_dependent , tokens


def lemma_form_add(new_data,tokens):
    modified_dependent = []
    for relation_dep in new_data:
        for token_dict in tokens:
            if relation_dep['dependentGloss'] == token_dict['originalText']:
                relation_dep['lemma'] = token_dict['lemma']
                modified_dependent.append(relation_dep)
    return modified_dependent

def parse_sub_tree(new_data,dependent_dict,sub_dependent):
    word = dependent_dict['lemma']
    sub_dependent.append(word)
    dependent_index = dependent_dict['dependent']
    for relation_dict1 in new_data:
        if relation_dict1['governor'] == dependent_index:
            parse_sub_tree(new_data ,relation_dict1,sub_dependent)
    return sub_dependent

def filters_tag(modified_dependent):
    #print ("Modified Dependent:::" , modified_dependent)
    dependent_list = [relation_dict for relation_dict in modified_dependent
                            if relation_dict['dep'] not in ['det','aux','auxpass','case','punct','cop','mark','neg']]
    #print ("Dependent list:::" , dependent_list)
    return dependent_list

def parse_dependency_tree(new_data,augmented_argument_structure ,root_flag=False):
    root_dependent = ''
    for relation_dict in new_data:
        if relation_dict['governor'] == 0 and relation_dict['governorGloss'] == 'ROOT':
            #root_node_name = relation_dict['dependentGloss']
            root_node_name = relation_dict['lemma']
            augmented_argument_structure.append(root_node_name)
            root_flag = True
            main_root = root_node_name
            dependent = relation_dict['dependent']
            root_dependent = dependent
            root_flag = True
            for new_relation_dict in new_data:
                if new_relation_dict['governor'] == root_dependent:
                    sub_relations = parse_sub_tree(new_data,new_relation_dict,[])
                    if len(sub_relations) ==1:
                        augmented_argument_structure.append(sub_relations[0])
                    else:
                        augmented_argument_structure.append((sub_relations))
    return augmented_argument_structure

def compute_similarty_score(model_word2vec,model_doc2vec,aug_argument_lists):

    sent_argument1 = aug_argument_lists[0]
    sent_argument2 = aug_argument_lists[1]
    print('This is augumented list :::',aug_argument_lists)
    #sent_argument1 = ["physician", "assistant"]
    #sent_argument2 = ['unable', 'printer', 'connect-system']
    matrix = np.zeros((len(sent_argument1), len(sent_argument2)))
    #matrix1 = np.zeros((len(sent_argument1), len(sent_argument2)))

    for index1 , word1 in enumerate(sent_argument1):
        for index2, word2 in enumerate(sent_argument2):
            v1 = 0
            v2 = 0
            if word1.find('-') > -1:
                v1 = model_doc2vec.infer_vector([word1])
            else:
                #print (word1)
                try:
                    v1 = model_word2vec.wv[word1]
                except:
                    pass

            if word2.find('-') > -1:
                v2 = model_doc2vec.infer_vector([word2])
            else:
                #print (word2)
                try:
                    v2 = model_word2vec.wv[word2]
                except:
                    pass
            v1 = v1.reshape(1,-1)
            v2 = v2.reshape(1,-1)
            matrix[index1,index2] = cosine_similarity(v1,v2)[0][0]
            #matrix1[index1,index2] = np.linalg.norm(v1-v2)

    print ("\n Matrix :::\n\n",matrix)
    norm_matrix = normalize(matrix, norm='l2', axis=1, copy=True, return_norm=False)
	
    print("Normalized matrix is ::",norm_matrix)
    print("Normalized matrix Summation ::",norm_matrix.sum())
    print ("Score of the sentence:::::\n\n" ,matrix.sum())
    #data = matrix.max(axis=1)
    #print ("distance Score of the sentence:::::" ,matrix1.sum())

def load_word_2_vec():
    with open('word2_vec.pickle', 'rb') as handle:
        model = pickle.load(handle)
    return model

def load_doc_2_vec():
    with open('d2v.model', 'rb') as handle:
        model = pickle.load(handle)
    return model


def sentence_similarty_handler(sent1,sent2):
    #### Load Word2 Vec Model ######
    sent1 = sent1.lower()
    sent2 = sent2.lower()
    model_word2vec = load_word_2_vec()
    model_doc2vec  = load_doc_2_vec()

    aug_argument_lists = []
    for sent in [sent1,sent2]:
        basic_dependent,tokens = get_parser_output(sent)
        modified_dependent = lemma_form_add(basic_dependent,tokens)
        #print('This is modified dependent output of lemma ::',modified_dependent)
        dependent_list = filters_tag(modified_dependent)
        print('***'*20)
        print(dependent_list)
        print('***'*20)
        continue
        #print ("DEpenedent:::" ,dependent_list)
        augmented_argument_structure = []
        augmented_argument_structure = parse_dependency_tree(dependent_list,augmented_argument_structure)
        #print ("Augmented Argument Structure1:::" , augmented_argument_structure)
        new_augmented_structure = []
        for parse_word in augmented_argument_structure:
            if type(parse_word) == type([]):
                new_augmented_structure.append("-".join(parse_word))
            else:
                new_augmented_structure.append(parse_word)
        print ("Augmented Argument Structure2:::" ,new_augmented_structure)

        aug_argument_lists.append(new_augmented_structure)
    compute_similarty_score(model_word2vec,model_doc2vec,aug_argument_lists)


sent1 = 'The earnings of a company are determined by its operating costs financing assets and liabilities.'
sent2 = 'A balance sheet is a financial statement that shows a company’s level of assets liabilities and shareholders equity.'



#sent1 = 'network is working.'
#sent1 = 'printer configure'
#sent2 = 'printer is able to connect.'
#sent2 = 'The little Jerry is being chased by Tom in the big yard.'
#sent2 = "Printer is not working."
#sent2 = "Printer is unable to connect by system."
#sent2 = "To compare two strings you can use"
#sent2 = "cab is not available"
#sent1 = "How old are you?"
#sent2 = "What is your age?"
sentence_similarty_handler(sent1,sent2)
