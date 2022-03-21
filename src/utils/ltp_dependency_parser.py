from pyltp import Parser, Segmentor, Postagger, SentenceSplitter, NamedEntityRecognizer
import os
import csv
from file_operation import XMLOperation
import json


# 加载模型路径
LTP_DATA_DIR = 'external_tools/ltp_model/ltp_data_v3.4.0'
PARSER_MODEL_PATH = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径
SEGMENTOR_MODEL_PATH = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径
POS_MODEL_PATH = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径
NER_MODEL_PATH = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`

# 加载命名实体识别模型
ner_parser = NamedEntityRecognizer()
ner_parser.load(NER_MODEL_PATH)  # 加载模型
# 加载依存句法分析模型
parser = Parser()
parser.load(PARSER_MODEL_PATH)
# 加载词性标注模型
postagger = Postagger()
postagger.load(POS_MODEL_PATH)
# 加载分词模型
segmentor = Segmentor()
segmentor.load(SEGMENTOR_MODEL_PATH)

def get_ner_result(tokenization_list, postag_list):
    """
         得到命名识别结果
    :param tokenization_list:
    :param postag_list:
    :param ner_parser:
    :return:
    """

    return list(ner_parser.recognize(tokenization_list, postag_list))



def get_dependency_parser_result(tokenization_list, postag_list):
    """
        根据传入的分词和词性标注结果对句子做依存分析，并返回结果
    :param tokenization_list: 分词结果[list]
    :param postag_list: 词性标注结果[list]
    :param parser: 依存分析模型
    :return: 依存分析结果
    """

    arcs = parser.parse(tokenization_list, postag_list)

    return list(arcs)



def get_postag_result(words_list):
    """
        对传入的已经经过分词的句子做词性标注
    :param words_list: 经过分词得到的句子的词汇表[list]
    :param postagger: 词性标注模型
    :return: 词项标注结果[list]
    """

    postags = postagger.postag(words_list)

    return list(postags)



def get_tokenization_result(sentence):
    """
        对输入的原始文本做分词，并返回结果
    :param sentence: 原始句子[str]
    :param segmentor: 分词模型
    :return: 经过分词后的词语列表[list]
    """
    words = segmentor.segment(sentence)

    return list(words)


# def get_verbs_index_list(postags):
#     """
#         从词性标注的结果中得到所有的动词对应的下标
#     :param postags: 句子词性标注的结果[list]
#     :return: 动词对应的下标列表[list]
#     """
#
#     verbs_list = []
#
#     for index, postag in enumerate(postags):
#         if postag == 'v':
#             verbs_list.extend([index])
#
#     return verbs_list

def get_hed_verbs(arcs, postags):
    """
        得到核心动词对应的下标列表
    :param arcs: 依存分析结果
    :param postags: 词性标注的结果
    :return: 核心动词对应的下标列表
    """
    hed_verbs = []

    # 先找第一个和root相联的核心动词HED关系,enumerate函数从0开始迭代
    cur_index = 0
    for index, arc in enumerate(arcs):
        if arc.head == 0 and arc.relation == 'HED' and postags[index+1] == 'v':
            hed_verbs.extend([index+1])
            cur_index = index + 1
            break

    # 根据得到的第一个核心动词，得到后续和他并列的核心动词

    while True:
        has_coo = 0  # 存在这个关系
        for tmp_index, tmp_arc in enumerate(arcs):
            if tmp_arc.head == cur_index and tmp_arc.relation == 'COO' and postags[tmp_index+1] == 'v':
                hed_verbs.extend([tmp_index+1])
                has_coo = 1
                cur_index = tmp_index + 1
                break
        if has_coo == 0:
            break

    return hed_verbs

def get_event_based_on_hed_verb(hed_verb, arcs):
    """
        根据中心词返回动词事件中七元组对应分词序列中的下标位置，比如在识别宾语时，如果“苹果”在tokenization_list中的下标为7，那么宾语对应的返回值为[7]
    :param hed_verb: 核心动词下标
    :param arcs: 依存分析树
    :return: 返回七元组中每个位置对应的下标序列，如[[1], [2, 3], [4], [5, 6, 7,], ……](之所以返回这个值，是为了后面的template-matching方便，根据下标可以直接替换)
    """

    seven_tuple = {}  # 七元组定以为一个字典类型，元素分别为Att_Subj, Subj, Adv, P, Cpl, Att_Obj, Obj
    type = ['Att_Subj', 'Subj', 'Adv', "P", 'Cpl', 'Att_Obj', 'Obj']

    # 将谓语动词加入
    seven_tuple['P'] = [hed_verb]

    # 得到主语部分
    has_subj = 0  # 标识是否有主语
    for index, arc in enumerate(arcs):
        if arc.head == hed_verb and arc.relation == 'SBV':
            seven_tuple['Subj'] = [index+1]
            has_subj = 1
            break
    if has_subj == 0:
        seven_tuple['Subj'] = ['#Token']
        seven_tuple['Att_Subj'] = ['#Token']  # 没有主语则肯定没有修饰成分
    else:  # 如果有主语，接着判断是否有主语修饰成分
        subj_index = seven_tuple['Subj'][0]
        has_att_subj = 0
        for index, arc in enumerate(arcs):
            if arc.head == subj_index and arc.relation == 'ATT':  # 如果存在修饰成分
                tmp_index = index + 1
                # 则再判断是否存在ATT关系的迭代形式
                while True:
                    has_att = 0  # 标记当前是否存在ATT的迭代，找到最左端的位置

                    for i, tmp_arc in enumerate(arcs):
                        if tmp_arc.head == tmp_index and tmp_arc.relation == 'ATT':  # 如果存在迭代
                            tmp_index = i + 1
                            has_att = 1
                            break
                    if has_att == 0:
                        break

                seven_tuple['Att_Subj'] = list(range(tmp_index, subj_index))
                has_att_subj = 1
                break
        if has_att_subj == 0:
            seven_tuple['Att_Subj'] = ['#Token']

    # 得到直接宾语部分
    has_obj = 0  # 标识是否有宾语
    for index, arc in enumerate(arcs):
        if arc.head == hed_verb and (arc.relation == "VOB" or arc.relation == 'FOB'):

            # 判断是否有双宾语
            has_double_obj = -1
            for tmp_index, tmp_arc in enumerate(arcs):
                if tmp_arc.head == index + 1 and tmp_arc.relation == 'VOB':
                    seven_tuple['Obj'] = [index+1, tmp_index+1]
                    has_double_obj = 1
                    break

            if has_double_obj == -1:  # 如果没有双宾语
                seven_tuple['Obj'] = [index+1]
            has_obj = 1
            break
    if has_obj == 0:
        seven_tuple['Obj'] = ['#Token']
        seven_tuple['Att_Obj'] = ['#Token']  # 如果没有宾语，则必定没有修饰成分
    else:  # 如果有直接宾语，那么判定有没有Att_Obj部分
        Obj_index = seven_tuple['Obj'][0]
        has_att_obj = 0
        for index, arc in enumerate(arcs):
            if arc.head == Obj_index and arc.relation == 'ATT':
                tmp_index = index + 1
                # 判断修饰成分是否存在迭代
                while True:
                    has_att = 0

                    for i, tmp_arc in enumerate(arcs):

                        if tmp_arc.head == tmp_index and tmp_arc.relation == 'ATT':
                            tmp_index =  i + 1
                            has_att = 1
                            break
                    if has_att == 0:
                        break


                seven_tuple['Att_Obj'] = list(range(tmp_index, Obj_index))  # 修饰部分得到的是一个区间
                has_att_obj = 1
                break
        if has_att_obj == 0:
            seven_tuple['Att_Obj'] = ['#Token']

    # 找到Adv部分，当为介宾关系时，以POB结束
    has_adv = 0
    start_adv = -1
    for index, arc in enumerate(arcs):  # 先确定是否有Adv
        if arc.head == hed_verb and arc.relation == 'ADV':
            start_adv = index + 1  # 先找到Adv的起始位置
            has_adv = 1
            break
    if has_adv == 1:  # 如果有Adv
        end_adv = -1
        for index, arc in enumerate(arcs):  # 寻找是否有POB关系
            if arc.head == start_adv and arc.relation == 'POB':
                end_adv = index + 1
                break
        if end_adv > -1:
            seven_tuple['Adv'] = list(range(start_adv, end_adv+1))
        else:
            seven_tuple['Adv'] = list(range(start_adv, hed_verb))

    elif has_adv == 0:
        seven_tuple['Adv'] = ['#Token']

    # 找到Cpl部分
    has_cpl = 0
    for index, arc in enumerate(arcs):
        if arc.head == hed_verb and (arc.relation =='CMP' or arc.relation == 'DBL' or arc.relation == 'IOB' or arc.relation == 'RAD'):

            if arc.relation == 'CMP':  # 如果有POB结尾，则为一个介宾结构，不止包含一个元素下标，所以需要判断这种情况
                end_cpl = -1
                tmp_index = 0
                for tmp_index, tmp_arc in enumerate(arcs):
                    if tmp_arc.head == index + 1 and tmp_arc.relation == 'POB':
                        end_cpl = tmp_index + 1
                        break
                if end_cpl > -1:  # 说明存在POB介宾结尾
                    seven_tuple['Cpl'] = list(range(index+1, tmp_index+2))
                else:
                    seven_tuple['Cpl'] = [index+1]
            else:  # 如果为其他关系
                seven_tuple['Cpl'] = [index+1]
            has_cpl = 1
            break
    if has_cpl == 0:
        seven_tuple['Cpl'] = ['#Token']

    return seven_tuple


def get_component(words_index, tokenization_list):
    """
        将一个词语下标序列转化为对应的中文短语
    :param words_index: 词语下标序列
    :param tokenization_list: 分词结果
    :return: 对应的中文短语
    """
    phrase = ''

    if words_index[0] == '#Token':  # 不存在这个成分时
        return phrase
    for index in words_index:
        phrase += tokenization_list[index]

    return phrase


def indexs2words(seven_tuple, tokenization_list):
    """
        根据事件七元组的下标序列，将每个事件七元组转化成中文短语
    :param seven_tuple: 事件七元组
    :param tokenization_list: 分词结果列表
    :return: 将下标序列转化为对应的中文短语后的七元组和完整的动词事件字符串
    """
    event = ''
    seven_tuple_in_chinese = {}

    # Att_Subj
    att_subj_phrase = get_component(seven_tuple['Att_Subj'], tokenization_list)
    seven_tuple_in_chinese['Att_Subj'] = att_subj_phrase
    event += att_subj_phrase

    # Subj
    subj_phrase = get_component(seven_tuple['Subj'], tokenization_list)
    seven_tuple_in_chinese['Subj'] = subj_phrase
    event += subj_phrase

    # Adv
    adv_phrase = get_component(seven_tuple['Adv'], tokenization_list)
    seven_tuple_in_chinese['Adv'] = adv_phrase
    event += adv_phrase

    # P
    p_phrase = get_component(seven_tuple['P'], tokenization_list)
    seven_tuple_in_chinese['P'] = p_phrase
    event += p_phrase

    # Cpl
    cpl_phrase = get_component(seven_tuple['Cpl'], tokenization_list)
    seven_tuple_in_chinese['Cpl'] = cpl_phrase
    event += cpl_phrase

    # Att_Obj
    att_obj_phrase = get_component(seven_tuple['Att_Obj'], tokenization_list)
    seven_tuple_in_chinese['Att_Obj'] = att_obj_phrase
    event += att_obj_phrase

    # Obj
    obj_phrase = get_component(seven_tuple['Obj'], tokenization_list)
    seven_tuple_in_chinese['Obj'] = obj_phrase
    event += obj_phrase

    return event, seven_tuple_in_chinese


def extract_events_from_sentence(arcs, postags, tokenization_list):
    """
        对进行了依存分析的句子根据其依存分析和词性标注结果抽取事件
    :param arcs: 依存分析的结果
    :param postags: 词性标注的结果
    :param tokenization_list: 分词结果
    :return: 一个事件列表，列表中的每个元素为一个抽取事件的字符串[seven_tuple, seven_tuple_in_chinese]
    """
    # 依存分析的结果是从1开始计数的，但依存分析树中并没有保存root节点，为了对齐，在分词和词性标注序列的起始位置都加入一个填充元素
    tokenization_list.insert(0, 'root')
    postags.insert(0, '0')

    # for arc in arcs:
    #     print("{}:{}".format(arc.head, arc.relation))

    # 首先得到所有的核心动词对应的下标
    print("7------")
    hed_verbs = get_hed_verbs(arcs, postags)
    print("5-------")
    # for index in hed_verbs:
    #     print("-----", tokenization_list[index])

    events = []  # 动词事件列表
    # 对每个动词抽取其的动词事件
    for hed_verb in hed_verbs:
        seven_tuple = get_event_based_on_hed_verb(hed_verb, arcs)
        # print("************************************************")
        # print(seven_tuple)
        event, seven_tuple_in_chinese = indexs2words(seven_tuple, tokenization_list)

        # print(event, "-----------------", seven_tuple_in_chinese)

        events.append([seven_tuple, seven_tuple_in_chinese])
    print("6------")
    return events


def add_subj_to_events(events):
    """
        如果所有的动词事件中只包含了一个主语，则将这个主语作为所有动词事件共同的主语
            这样做的目的是为了在避免冗余的前提下尽可能多的利用句子的信息抽取动词事件
    :param events: 事件列表
    :return: 返回经过了处理的事件列表
    """
    # 先判断是否只有一个主语
    subj_count = 0
    subj = []
    for event in events:
        if event[0]['Subj'] == '#Token':  # 当前动词事件没有主语
            continue
        else:  # 当前动词事件存在主语
            if subj_count < 1:  # 说明还没找到主语
                subj.append(event[0]['Subj'])
                subj.append(event[1]['Subj'])
                subj_count += 1
            else:
                subj_count += 1  # 说明有多个主语，此时加1是为了当做标志变量
                break
    if subj_count == 1:  # 只有只存在一个主语时，才将这个主语作为所有动词事件的主语
        for event in events:
            event[0]['Subj'] = subj[0]
            event[1]['Subj'] = subj[1]

    return events


def get_samples(csv_file_path, xml_file_path):
    """
        读取数据，对每个样本进行动词事件七元组的抽取
    :param file_path: 数据文件路径
    :return: None
    """

    print("Start processing data……")

    with open(csv_file_path, 'r', encoding='UTF-8') as file_handler:

        # print('1-------------------')

        file_reader = list(csv.reader(file_handler))
        # print("4--------{}".format(len(list(file_reader))))
        xml_writer = XMLOperation(xml_file_path)
        # print("5--------{}".format(len(list(file_reader))))
        # progress_bar = ShowProcess(len(list(file_reader)), 'Events Extraction Finished !!')

        # print("3-------------------{}".format(len(list(file_reader))))

        for index, sample in enumerate(file_reader):

            print("sample:{}".format(index+1))

            # 首先我们需要对样本进行分句
            # sents = SentenceSplitter.split("星期六我在家里和我的兄弟(双胞胎)吵架了。其中一个在私立学校学习，另一个在上五年级。他们不知道自己在做什么。他们遵循老师的每一个字，书籍和系统。他们总是试图通过一些游戏来为自己的不良行为辩解，他们对自己尴尬的生活感到满意。我多次试图说服他们，但都失败了。这次我打了一架，因为他们没有自愿精神(他们没有擦地板)。")
            # sents = SentenceSplitter.split("我在作业中发现了一个问题，虽然我尽了最大的努力，但还是没有解决。")
            sents = SentenceSplitter.split(sample[0])

            total_events = []  # 所有抽取的动词事件
            # 对每个句子进行处理
            for sentence in sents:

                # print("2--------------")

                # 分词
                tokenization_list = get_tokenization_result(sentence)
                # print(tokenization_list)
                # 词性标注
                postags = get_postag_result(tokenization_list)
                # print(postags)
                # 依存分析
                arcs = get_dependency_parser_result(tokenization_list, postags)

                events = extract_events_from_sentence(arcs, postags, tokenization_list)

                """
                    每个句子中可能存在多个动词事件，每个动词事件为一个元素，每个动词事件又由两个元素组成：一个为七元组都用
                        下标表示(下标对应分词结果列表)seven_tuple，一个为七元组都用相应的中文短语表示seven_tuple_in_chinese，
                            一个事件的结构为[seven_tuple, seven_tuple_in_chinese]
                    即一个句子对应的事件列表结构为[[seven_tuple, seven_tuple_in_chinese], [seven_tuple, seven_tuple_in_chinese], ……]    
                """
                events = add_subj_to_events(events)  # 判断是否只有一个主语，如果是的话，对主语进行填充

                total_events.extend(events)

                # print(events)
            print("Start to write XML……")
            xml_writer.write_events_to_xml([sample[0], sample[1], index+1], total_events)

            # progress_bar.show_process(index+1)  # 进度条实时显示事件抽取进度

        xml_writer.save_xml()


if __name__ == "__main__":

    get_samples(csv_file_path="./dataset/isear.dataset.chinese.v1.csv", xml_file_path="./dataset/ImplicitECD.Tuple.forTag.xml")
    # print(get_tokenization_result("今天天气真好。"))
