#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# In this lesson we are going to talk about using code to realize word2vec.
# 代码参考了DerekChia老师在github上的一个项目，分三部分，
# 第一部分，读取数据、构造单词Embedding和One-Hot向量。
# 第二部分，根据word2vec的思想构造训练数据。
# 第三部分，构造前向/后向传播函数和softmax函数对数据进行训练。
# 第四部分，获取词向量，构造相似度函数，计算单词之间的相似度。
# DerekChia老师这个项目里的源代码主要使用了Python的类，我把它修改成了许多函数，这样运行和看起来会比较方便。
"""
Created on Fri Jul 31 13:40:40 2020
    
    Word2Vec代码原理详解
    Ref: https://github.com/DerekChia/word2vec_numpy

@author: sunday
"""

# 我们导入了两个模块，一个是大家比较熟悉的numpy，另一个是从collections中导入了defaultdict这个方法
# 这是一个构造数据字典的方法，具体的使用方式我们一会儿会进行介绍
import numpy as np
from collections import defaultdict

np.random.seed(1234)

#########################--第一部分，读取数据、构造单词Embedding和One-Hot向量。
# 1.读取数据
def read_data():
    # 首先，第一小节，读取数据，我们先定义两个字符串，也就是两句话，作为输入
    # text1——自然语言处理和机器学习非常有趣，text2——自然语言处理和python是非常有意思的。
    # 然后把两句话放到一个列表里，如果我们有更多数据，texts这个列表就保存了许多句的文本内容。现在这里只有两个。
    text1 = "natural language processing and Machine learning is fun and exciting"
    text2 = "natural language processing and Python is nice and exciting"
    
    texts = [text1, text2]
    # 然后我们新建一个corpus列表，这个列表首选循环所有texts列表中的所有句子
    # 对于每句话，也就是每个text，对其做for循环将所有单词按空格分割开，再把所有的大写单词转化为小写，保存在一个列表中。
    # 所以我们看这些corpus这个列表，里面包含了两个列表，每个列表是一句话，里面的每个列表就是每句话的所有转成小写的单词。
    corpus = [[word.lower() for word in text.split()] for text in texts]
    
    return corpus

# 我们现在可以定义和调用这个read_data函数，得到corpus数据。
corpus = read_data()

# 2.构造Embedding
# 第二小节，我们来看怎么构造单词的Embedding
# 这里的Embedding是狭义上的Embedding，可以认为是一种唯一索引的Embedding
# 来看一下它的构造过程
# 首先，我们调用之前从collections导入的defaultdict模块，定义一个defaultdict对象word_counts
# 来看一下，word_counts现在是一个defaultdict对象，具有int属性，取值为空字典
# 接下来，我们循环corpus列表，对于它的每个元素，也就是每个列表，再进行for循环，取其中的每个单词
# 然后让word_counts这个对象每次累加1
# 运行玩这些代码后，我们看一下word_counts现在变成了什么
# 可以看到，word_counts原来的空字典变成了这样一个字典
# 这个字典的意思呢，就是相当于把corpus里的所有数据
# 或者我们输入的文本的所有单词，统计了它们出现的个数
# 例如我们可以数一下，原来的两句话中有4个and
# 一个，两个，三个，四个，原来的两句话中有两个is，一个，两个，这里

word_counts = defaultdict(int)
for row in corpus:
    for word in row:
       word_counts[word] += 1

# 字典构造完成后，我们定义一个v_count，保存我们这个字典中的所有key的个数
# 实际上，由于我们的字典word_counts保存的是所有数据的唯一单词及其出现的个数
# 那么v_count相当于整个字典的长度，或者说所有唯一单词的总个数
# 我们来看一下，结果是11
v_count = len(word_counts.keys())

# 我们再定义一个words_set列表，保存作为所有唯一单词的集合
# 实际上，v_count的大小就是words_set的长度
# 可以看一下，words_set和word_counts这一列的取值是一样的
# 它的总单词数也是11
words_set = list(word_counts.keys())
print(len(words_set))

# 最后，我们在对words_set这个列表进行for循环
# 对列表使用enumerate函数再进行for循环，能够把每个元素的值和它在列表中的index取出来
# 我们的word_index在for循环时保存的是以单词word为键，以index，i为值的字典
# index_word正好相反，是以index，i为键，以单词word为值
# 我们运行两行代码分别看一下这两个字典的结构
# 实际上相当于这两列调换了位置，所以它们的名字也正好相反
word_index = dict((word, i) for i, word in enumerate(words_set))
index_word = dict((i, word) for i, word in enumerate(words_set))

# 3. 构造One-Hot向量
# 第三小节我们讲解如何构造One-Hot向量
# 我们定义这样一个word2onehot的函数，传入单词，ｖ_count和word_index三个变量
# 假设我们传入的单词是and，在这里，我们定义word="and"
# 首先，我们预定义一个word_vec的列表
# 它是根据v_count来定义的，也就是我们唯一单词的总数量来定义
# 我们看到，现在它是一个全0的列表，里面有v_count长度个0，现在也就是11个0
# 我们定义word_ix等于word_index这个字典中以传入的word为key的值
# 现在word="and"这个单词，在word_index这个字典中，我们发现它的值为３
# 也就是现在我们的word_ix=3，紧接着，我们令word_vec这个列表的第word_ix个元素，
# 也就是第４个元素，赋值为1
# 来看一下现在这个word_vec，现在它在第四个位置上取值为１，其他元素取值为０
# 有同学有些疑问为什么不是第３个位置取值为１，因为我们word_index字典的索引是从０开始取的
def word2onehot(word, v_count, word_index):
    # word = 'and'
    word_vec = [0 for i in range(0, v_count)]
    word_ix = word_index[word]
    word_vec[word_ix] = 1
    return word_vec

# 因此，我们的word2onehot函数的作用是什么呢，
# 当我们传入一个单词，根据这个单词在我们所维护的word_embedding，也就是word_index字典中的位置，
# 需要我们输出一个向量，向量的长度等于word_embedding字典的长度
# 向量中的元素，除了我们传入的单词在对应位置上为１，其他位置全部为０
# 再比如这里，假设我们传入的是fun这个单词，
# 我们看到，输出的word_vec在第８个单词的位置取值为１，其他取值为０
# 可以想象一下，如果我们的语料库很大，不重复单词的个数很多，
# 多到类似于我们在现实中买到的格林英语词典中的单词这么多，
# 实际上我们的word_index或者index_word就是对这个词典中的所有单词构建了一个python字典
# 假设有20000个单词，那么我们的one-hot向量的长度就是20000
# 相当于每个单词都有一个自己的one-hot向量，这样如果再给定一句话，
# 就是说一句话有多少个单词，就有多少个20000长度的one-hot向量，
# 那么这样，一句话就可以认为是多个向量组成的矩阵了。
# 多句话就是多个矩阵，也就是三维张量的概念了。
# 这里不理解的同学可以再仔细体会一下。

# word = "fun"
# word_vec = word2onehot(word, v_count, word_index)
# print(len(word_vec))

#########################--第二部分，根据word2vec的思想构造训练数据。

# 讲完这些内容后，我们讲解第二部分，根据word2vec的思想构造训练数据。
# 首先来回顾一下word2vec构造周围词和目标词的方式，
# 在CBOW模型中，输入是某个单词的周围词，输出目标词是周围词中间的当前词。
# 在Skip-Gram模型中，情况正好相反。
# 我们定义get_training_data这样一个函数，传入三个参数
# corpus,v_count和word_index，都是我们之前比较熟悉的
# 首先，定义一个空列表training_data
# 对corpus中的每句话——也就是每个列表进行循环
# 定义循环到的这句话的长度为sent_len
# 这里我们以corpus的第一个元素为例，sentence就是我们现在这个列表
# 现在这句话里有10个单词，因此我们的sent_len＝10
# 继续对这句话的每个单词进行循环，我们使用enumerate函数同时取出循环到单词的索引i和单词word
# 为了方便讲解，我们也假设取i=3,也就是word=“and”时的数据
# 我们定义w_target目标词是sentence的第i+1个单词（索引从0开始），
# 也就是“and”这个单词，使用word2onehot函数，得到and这个单词的one-hot向量作为w_target
# 可以看到，w_target这个结果是我们刚刚见过的。
# 然后我们定义一个周围词或者上下文词的空列表，
# for循环是从i-2一直循环到i+2+1,注意到我们现在的i是等于3的，
# 就是说，for循环是从3-2=1循环到3+2+1=6,就是1到6
# 我们打印出每个j的结果，可以看到是1/2/3/4/5
# 需要注意的是，这里的i-2和i+2+1里的2，指的是word2vec滑窗技术中的窗口的大小，后面我们会再介绍
# 紧接着，我们的for循环下面有个if条件进行约束，
# 只有当j大于等于0，j不等于i并且j小于句子长度sent_len-1时才进行下面的操作
# 注意到，现在我们的i是等于3的。所以在if条件的限制下，就只有1,2,4,5了，排除了3的情况，
# 我们再打印一下，看一下结果，1,2,4,5
# 在i取值小于2和大于最大长度-2时，情况有所不同，
# 假设我们的i=0, 如果不加if条件，我们先打印一下j，可以看到取值为-2,-1,0,1,2
# 加上if条件的限制，我们发现除了排除了当前的i=0，j>=0的限制也排除了-2,-1。因此只剩下了1,2
# 为了更加清晰地理解这种滑窗构造数据的过程，我们打开DerekChia老师的github或者博客，
# 我们看到这里第一张表，从上往下一共有10行，也就是我们代码中的第二层for循环的次数，
# 或者说我们句子中所有单词的总个数。
# 我们来看第一行，这里就是i=0的情况，也就是我们的w_target现在是natual这个单词
# 刚刚我们分析过，i=0时，第三层for循环加上if条件的限制，最终只剩下1和2，
# 在这个图中，也就是我们的后面两个单词language和processing
# i=3时，也就是来看第4行，and单词作为w_target，
# 周围的四个单词language、processing,mechine和learning作为周围词w_context
# i=1, i=2, i=3，一直到最后，每次循环构造的数据就如图所示。
# 继续回到我们的代码中，我们发现当if条件满足时，
# w_context就定义为所有符合条件的句子中的第j个词的one-hot向量，
# 那么由于我们的w_contexts，是在第二层for循环运行到第i个单词的时候定义的
# 第三层for循环可以说是根据i来构造的周围词数据，
# 因此w_contexts在第三层for循环时append每一个周围词w_context（末尾少了个s）
# 所以可以看到，我们一个i对应一个w_target,对应一个w_contexts，也就是一组周围词
# 我们的training_data在每次完成第i词循环以后，同时append了w_target和w_contexts
# 加上第一层for循环，所以最终training_data保存了所有句子中的所有当前词和周围词的数据
# 最后用np.array将training_data列表转换成了numpy的array数据
# 我们定义函数并整体调用一下，看一下traning_data最后的结构
# 首先，我们发现一共有19条数据，对应于我们原始输入数据中的两句话所有单词的总个数。
# 打开看一下，第一列就是所有这19个单词每个单词的one-hot向量。
# 第二列，对应于第一列的每一个单词，就是这个单词的滑动窗口为2的周围词的one-hot向量
# Ok，到这里的话，我们的训练数据就构造好了。
# 关于训练数据one-hot的表示如果大家有不太懂的也可以结合DerekChia老师的第二张图来进行理解
# 第二张图与第一张展示的本质内容是一样的，区别是第二张图使用了更详细的one-hot向量图进行了展示
# 这一小节的视频我们就讲这么多，下一节我们将继续讲解后面的内容。谢谢大家。

def get_training_data(corpus, v_count, word_index):
    
    training_data = []
    for sentence in corpus:
        # sentence = corpus[0]
        sent_len = len(sentence)
        for i, word in enumerate(sentence):
            # i = 1; word = 'and'
            w_target  = word2onehot(sentence[i], v_count, word_index)
            w_contexts = []
            for j in range(i - 2, i + 2 + 1):
                # print(j)
                if j >= 0 and j != i and j < (sent_len-1):
                    # print(j)
                    w_context = word2onehot(sentence[j], v_count, word_index)
                    w_contexts.append(w_context)
                    
            training_data.append([w_target, w_contexts])
           
    return np.array(training_data, dtype=object)

training_data = get_training_data(corpus, v_count, word_index)

#########################--第三部分，构造前向/后向传播函数和softmax函数对数据进行训练。

# （tongxuemen，dajiahao，zhejiekewomen继续jiangjieword2vec后面的内容
# 我们暂时先不讲这三个函数，softmax,forward_prop和backward_prop。
# 实际上，在上节课构造完数据后，我们第一时间考虑的事情就应该是训练了。
# 那么这三个函数就是我们训练过程中所要用到函数，我们会在用到的时候再讲。
# 



# 在构造训练数据之前，我们需要先初始化一些模型参数和定义一些超参数来用。
# 首先，我们使用numpy定义w1和w2两个模型的初始化参数，
# 大家可以把我们的任务简单理解为 y = x*w1*w2 
# 这里，y可以理解为我们的w_context, x可以理解为我们的w_target
# 也就是我们training_data的两列数据，那么w1，w2是最终想要得到的参数数据，
# 我们知道，不论是使用神经网络还是传统机器学习，学习参数的过程都是根据给定的x，y
# 先初始化一组随机的[w1, w2]，然后计算出一个y*，根据y*和y的差异(loss)，和梯度下降的原则，
# 反向调整我们的[w1, w2]得到一组新的[w1~, w2~]，再输入进去得到另一个y*去跟y进行比较，
# 这样一次一次的迭代，就完成了我们对参数[w1, w2]的更新，在一定的准确率条件或者loss下降到一定程度下，
# 最终获得的[w1, w2]就是我们需要的“模型”。

# 注意到，我们现在定义的w1和w2维度分别是(11, 100)和(100, 11),设置这些维度的目的我们待会儿再讲
# 首先，我们固定一个numpy取随机数的seed为1234，
# 这样不管我们运行多少次这个代码文件，每次的随机数取值都是一样，
# 然后我们定义第一个超参数-学习率lr为0.01，学习率的作用是决定更新模型参数w1，w2的变化程度
# 定义迭代次数为100，我们需要对所有数据进行多次迭代才能收敛到合适的loss

# 2. 初始化权重矩阵：
w1 = np.random.uniform(-1, 1, (len(index_word), 100)) # print(w1.shape)
w2 = np.random.uniform(-1, 1, (100, len(index_word))) # print(w2.shape)
# 3. 定义训练所需的超参数
lr = 0.01
epochs = 100


# 4.训练

# 设置完这些参数和超参数以后，现在来定义训练函数，
# 我们在train函数这里传入training_data, w1, w2, lr, epochs这几个参数
# 训练过程首先对100次迭代进行循环，我们设置epoch=0,loss现在被赋值为0
# 然后对training_data进行for循环，分别获w_target和w_contexts
# （打开training_data）前面我们看过training_data的数据，for循环就是一行行读取它的数据
# 假设现在我们令w_t, w_c = training_data[3]，即假设for循环到第4行
# 注意到，w_t我们发现，就是我们熟悉的当前词and的one-hot向量
# w_c就是and单词的周围词，也就是language、processing,mechine和learning的one-hot向量
# 仔细看w_c这个矩阵我们发现，第一行是第二个元素为1，第二行是第三个元素为1
# 第三行是第五个元素为1，第四行是第六个元素为1，所以唯独少了第四个元素为1的数据
# 实际上少的这条数据就是w_t，也就是当前词或者目标词的one-hot向量

# 下一步，我们将取到的w_t这个one-hot向量和之前初始化好的w1,w2传入forward_prop函数中，
# 也就是进行前向传播操作，我们回去看一下forward_prop函数是怎样的
# 首先，forward函数的第一行计算了一个h=np.dot(w_t, w1)
# 我们可以归纳地写成 h=x*w1，同时，注意到w_t现在是一个列表，
# 其维度认为是(1, 11)维，w1的维度是(11, 100)维,从而得到h的维度是(1,100)维
# 继续看，u=np.dot(h, w2), 可以理解为 u = h*w2 = x*w1*w2
# 注意到h的维度是(1, 100)维，w2的维度是(100, 11)维,从而得到u的维度是(1, 11)维
# 最后，yp = softmax(u)，也就是可以写成 yp=softmax(x*w1*w2)
# 这里我们再上去看一下softmax函数，首先我们打印一下向量u看一下，u是一个长度为11的一维向量
# softmax函数对u进行了如下的操作：首先，将u的每个元素减去u中的最大的元素，
#（运行np.max(u)),注意到，np.max(u)得到u中11个元素的最大值是第三个元素2.09014041
# u - np.max(u)得到一个新的长度为11的一维向量，其中的元素为u逐元素减去u的最大值
# 特别注意一下，因为u中最大的元素是第三个，所以新的向量的第三个元素为0
# 接下来，我们使用np.exp()即以e为底的指数函数对u-max(u)的所有元素进行作用，
# 得到下面这个长度还是11的以为向量，特别注意到的是，第三个元素原来为0，
# 使用指数函数后，变成了1，也就是说 e^0 = 1，验证了我们的结果
# 现在的结果被保存为了e_u这个变量，注意到我们return的是e_u / e_u.sum(axis=0)
# 就是说返回的值是e_u除以e_u在第0维度的求和值，我们首先来看一下e_u.sum(axis=0)
# 运行得到1.946955，因为向量e_u的维度只有1维，所以这个求和项就是其中11个元素的求和
# 最后，用e_u除以这个求和项，实际上是对e_u的每个元素都除以这个和
# 我们来看，最后这个结果，e_u / e_u.sum(axis=0), 对这些结果求和
# 也就是运行sum(e_u / e_u.sum(axis=0)),发现其结果等于1，也就是说这些数值是一个概率分布
# 由于在forward_prop最后，yp是经过softmax后的最终结果。

# 我们的前向传播forward_prop函数最终返回了最后一层的结果yp, 第一层的结果h和第二层的结果u
# 在现在大部分网站中对word2vec的介绍中，通常把第一层看成是投影层，第二层是全连接层，
# 第三层是softmax层。我们现在这个前向传播，实际上和神经网络的前向传播不同，
# 不知道大家有没有看出有什么区别？有什么区别呢？
# 细心的同学发现了，实际上神经网络的每一层和下一层之间会有一个激活函数进行作用
# 例如我们这里的全连接层，如果在神经网络中就不是u = np.dot(h, w2)了
# 而应该是 u = sigmoid(np.dot(h, w2))或者 u = relu(np.dot(h, w2))，对吧？
# 所以严谨的说，word2vec不是真正的神经网络结构，只是说它的层级也是像神经网络一样一层接一层
# 回到我们的代码，得到yp, h和u之后，我们继续看下面的代码
# 首先这行代码的意思是，首先我们对w下划线c进行for循环，对其中的每个wc，让其跟yp使用np.substract进行作用
# 最后把每个元素作用完成后的列表转成np.array格式
# 首先我们需要注意，w_c是一个列表套四个列表的数据，也就是我们当前词的四个周围词的one-hot向量
# 所以这里的疑问就是np.substract是什么操作。
# 为了方便演示起见，我们取wc=w_c的第一个元素，即w_c[0]
# 首先来看一下yp，现在是一组小数，也就是我们softmax之后的概率分布
# wc呢，现在是[0, 1, 0, 0...]，第二个元素为1，其他都为0
# 我们单独运行np.substract(yp, wc), 发现得到了一个和yp非常相似的向量
# 注意到，二者的区别只在于yp的第二个元素变了，没错，变成了yp中原来的数据减去1
# 因此，np.substract(a, b)的作用就是得到a和b两个向量逐元素相减得到的数据
# 那么我们回来看这一行代码，wc_sub就是前向传播一次得到的结果yp 对w_c中的每个周围词的one-hot向量逐元素相减，
# 然后转成np.array格式的数据。现在我们运行完这一行看一下wc_sub
# 实际上它和w_c的格式是一样的，也是一个向量中嵌套了四个“差异向量”
# 其差异的意思就是这个向量也是“周围词”的概念，只不过它的每个向量都被yp进行了“作差”。
# 最后，e = np.sum(wc_sub, axis=0)，是对wc_sub在axis=0的维度进行求和。
# 我们运行这一行看一下结果，求和后e变成了一行向量。再打开wc_sub看一下这个矩阵
# 我们发现，实际上e就是wc_sub的每一行全部加到第一行的结果
# 这里我们需要注意，还记得我们之前讲过的神经网络的构造过程，我们在前向传播和反向传播中间是需要计算loss的
# （打开NN_byNumpy.py)，这里。但是注意到，现在word2vec的loss写在了backward_prob后面，
# 但是，我们还需要注意到的是，实际上这里的wc_sub和L实际上也是loss的意思。
# 我们知道，在一般情况下y=f(x)，则loss=1/2*sum[(y-y*)^2],
# 在我们的这个word2vec的模型中，自变量x可以认为是w_target也就是当前词
# 因变量y是周围词，也就是w_contexts, 或者说y这里就是每一个周围词wc
# 我们归纳前向传播的计算公式是 yp=softmax(u); u=h*w2; h=w_t*w1=x*w1;
# 那么为了进行反向传播的梯度计算，我们首先定义一个dLdu，衡量了当前词的预测值yp和每个上下文词wc之间的差异。
# 实际上，大家可以查阅资料，交叉熵损失函数Loss对yp=softmax(u)函数的梯度，
# 正好就是softmax函数的结果yp减去每个实际值yi
# 在help文件中，我们来看np.substract，是对两个向量进行了逐元素相减
# 回到代码这里，就是对当前词的预测值yp的向量和每个周围次wc的one-hot向量进行了逐元素相减
# 得到的dLdu，连同在前向传播中返回的h, 和当前词w_t，
# 参数矩阵w1,w2和学习率lr一起传入反向传播函数
# 反向传播的目的在于根据学习率和loss函数对每个参数w1和w2的梯度，来更新参数w1，w2
# 在反向传播这里，首先我们计算Loss对参数w2的梯度dLdw2
# 根据计算公式，L，也就是Loss对w2的偏导数等于L对u的偏导数，乘以u对w2的偏导数
# 现在，L对u的偏导数就是dLdu，由于u=h*w2，u对w2的偏导数就等于h
# 因此dudw2 = dLdu*dudw2 = dLdu*h，注意，这里的np.outer实际上和np.dot代表的意思差不多
# 二者指的都是向量或者矩阵相乘的意思，我们来看help中的例子
# 现在矩阵a的shape=(1, 3)，矩阵b的shape=(1, 4),直接使用np.dot是报错没法计算的，
# 因为计算不符合矩阵点乘的规则，而使用np.outer则可以无视这种规则，
# 可以认为是对a矩阵先进行了转置，然后再进行计算，或者认为对b中的元素使用a进行逐元素相乘
# 但不管怎么样，以后我们遇见np.dot或者np.outer，就表示对里面的两个元素进行相乘操作，
# 这样我们就可以专心研究它们之间的乘积关系，暂时不用计较它们在相乘时用的点乘还是逐元素相乘
# 接下来，我们计算L对w1的偏导数。注意到，从Loss到w1共经过了3次运算
# 分别是yp=softmax(u); u=h*w2; h=w_t*w1=x*w1;
# 因此L对w1的偏导数dLdw1=dLdu*dudh*dhdw1，其中dhdw1=x=w_t, dudh=w2
# 因此dLdw1 = dLdu*w_t*w2，在代码中，w2首先和dLdu转置相乘，再和w_t相乘，
# 这里有一个np.outer和一个np.dot，与我们的计算公式是一致的
# 求出L对w1和w2的偏导数以后，我们可以使用学习率lr乘以梯度，让w1和w2以这种速度进行更新，
# 也就是下面这里w1 = w1 - (lr * dLdw1); w2 = w2 - (lr * dLdw2)，最后返回更新后的w1和w2
# 至此，我们一次迭代的训练过程已经讲完了，可以发现，每个epoch循环所有训练数，每次将loss初始化为0
# 一次一次地对w1和w2进行更新，最后计算得出的loss是逐渐降低的，下降到91左右，就差不多收敛了

# 1. 定义训练所需的softmax、前向传播和后向传播函数
# softmax
def softmax(u):
    e_u = np.exp(u - np.max(u))
    return e_u / e_u.sum(axis=0)

# 前向传播
def forward_prop(w_t, w1, w2):
    h = np.dot(w_t, w1) # (1, 11)*(11, 100) > (1, 100)
    u = np.dot(h, w2)   # (1, 100)*(100, 11) > (1, 11)
    yp = softmax(u)     
    return yp, h, u

# 后向传播
def backward_prop(dLdu, h, w_t, w1, w2, lr):
    
    # dudw2 = h.copy()       # ∂u∂w2
    dLdw2 = np.outer(h, dLdu)  # (∂u∂w2,∂L∂u): (N_h, N_e)-->(10, 9)
    
    # dudh = w2.copy(); dhdw1 = w_t.copy
    dLdw1 = np.outer(w_t, np.dot(w2, dLdu.T)) # (9, (10,9)*(9,1))-->(9, 10)
    w1 = w1 - (lr * dLdw1)  # (9, 10) - (1*(9, 10))
    w2 = w2 - (lr * dLdw2)  # (10, 9) - (1*(10, 9))
    return  w1, w2

# 关于Loss的计算：
# 我们这里简单推到一下这里这个loss为什么会是这样的结果
# 首先，我们知道Loss的固定公式Loss=-log(Y|X),这里我们的X表示当前词w_target，
# Y表示周围词w_contexts，因此Loss=-logP(w_contexts|w_target),我们简化为-logP(c|w)
# 我们假设w现在等于c3,把四个周围词带入公式，得到-logP(c1,c2,c4,c5|c3)=-logP(c1,c2,c4,c5|w)
# 由于我们考虑的是在w的条件下周围词c1,c2,c4,c5的概率，也就是说我们研究是当前词w和周围词的关系，
# 因此可以认为，在这种研究角度下，周围词对w的条件概率是相互独立的。
# 或者说，四个周围词对当前词w的影响，等于它们各自对w的影响
# 所以我们有-logP(c1,c2,c4,c5|w)=-log[P(c1|w)*P(c2|w)*P(c4|w)*P(c5|w)]
# 继续利用对数函数的性质，logmn=logm+logn,得到下面这个公式
# 

# 模型训练
def train(training_data, w1, w2, lr, epochs):
    
    for epoch in range(epochs):
        # epoch = 0
        loss = 0
        for w_t, w_c in training_data:
            # w_t, w_c = training_data[3]; print(len(w_t))
            yp, h, u = forward_prop(w_t, w1, w2)
            
            loss += -np.sum([u[wc.index(1)] for wc in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
            # wc = w_c[0]
            dLdu = np.sum(np.array([np.subtract(yp, yi) for yi in w_c]), axis=0) # softmax函数
            
            w1, w2 = backward_prop(dLdu, h, w_t, w1, w2, lr)
            
        print("Epoch:", epoch, "Loss:", loss)

train(training_data, w1, w2, lr, epochs)


#########################--第四部分，获取词向量，构造相似度函数，计算单词之间的相似度。
# 获取词向量
def word_vec(word, word_index, w1):
    w_index = word_index[word]
    v_w = w1[w_index]
    return  v_w

# 5.计算相似度
def vec_sim(word, v_count, top_n, w1, word_index, index_word):
    
    # 获取当前词的词向量
    v_w1 = word_vec(word, word_index, w1)
    
    # 计算获取当前词与所有词的相似度集合
    word_sim = {}
    for i in range(v_count):
        v_w2 = w1[i]
        theta_sum = np.dot(v_w1, v_w2)
        theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
        theta = theta_sum / theta_den

        word = index_word[i]
        word_sim[word] = theta
    
    # 相似度结果排序
    words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
    for word, sim in words_sorted[:top_n]:
        print(word, sim)

word = "language"
vec = word_vec(word, word_index, w1)
vec_sim(word, v_count, 5, w1, word_index, index_word)
















