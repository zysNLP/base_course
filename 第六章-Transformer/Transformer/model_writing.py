#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:08:41 2021

    Ref:https://github.com/649453932/Chinese-Text-Classification-Pytorch

@author: sunday
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding, model_name):
        
        self.dataset = dataset
        self.embedding = embedding
        self.model_name = model_name
        
        """1.初始化路径相关的变量"""
        self.path_train = self.dataset + '/data/train.txt'                                # 训练集路径
        self.path_dev = self.dataset + '/data/dev.txt'                                    # 验证集路径
        self.path_test = self.dataset + '/data/test.txt'                                  # 测试集路径
        self.path_vocab = self.dataset + '/data/vocab.pkl'                                # 唯一词表路径
        self.path_save = self.dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 训练结果保存路径
        self.path_log = self.dataset + '/log/' + self.model_name                          # 训练log保存路径
        
        """2.初始化Embedding相关的变量"""
        self.embedding_pretrained = self.define_embedding_pretrained()  # 指定是否使用预训练好的词向量模型
        self.embedding_dim = self.define_embedding_dim()                # 指定Embedding的维度
        
        """3.初始化模型训练相关的变量"""
        self.device = self.device()                                     # 指定设备类型
        self.n_vocab = 0                                                # 词表大小，在运行时赋值        
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.max_seq_len = 32                                           # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率        
        self.dropout = 0.5                                              # Dropout
        self.require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.class_list = self.read_class_list()                        # 类别列表
        self.num_classes = len(self.class_list)                         # 类别数
        
        """4.初始化模型结构相关的变量"""
        self.model_dim = 300                                            # Encoder层的输入层维度 
        self.hidden_size = 1024                                         # Encoder层的隐藏层维度 
        self.num_heads = 5                                              # 多头注意力机制的注意力头个数
        self.num_encoders = 2                                           # Encoder层的个数
    
    def read_class_list(self):
        """读取所有分类标签"""
        return [x.strip() for x in open(self.dataset + '/data/class.txt', encoding='utf-8').readlines()]
    
    def define_embedding_pretrained(self):
        """定义是否使用预训练的词向量模型"""
        if self.embedding != 'random':
            return torch.tensor(np.load(self.dataset + '/data/' + self.embedding)["embeddings"].astype('float32'))
        else:
            return None

    def define_embedding_dim(self):
        """定义词向量维度"""
        if self.embedding_pretrained != None:
            return self.embedding_pretrained.size(1)
        else:
            return 300
    
    def device(self):
        """指定设备类型"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''Attention Is All You Need'''

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        # 1.定义Word Embedding层
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=config.embedding_pretrained, 
                                                          freeze=False)
        else:
            self.embedding = nn.Embedding(num_embeddings=config.n_vocab, 
                                          embedding_dim=config.embedding_dim, 
                                          padding_idx=config.n_vocab-1)
        
        # 2.定义Position Embedding层
        self.postion_embedding = Positional_Encoding(embedding_dim=config.embedding_dim, 
                                                     max_seq_len=config.max_seq_len, 
                                                     dropout=config.dropout, 
                                                     device=config.device)
        
        # 3.定义Encoder层
        self.encoder = Encoder(model_dim=config.model_dim, 
                               num_heads=config.num_heads, 
                               hidden_size=config.hidden_size, 
                               dropout=config.dropout)
        
        # 4.定义多个Encoder层
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(config.num_encoders)])

        # 5.定义全连接层
        self.fc1 = nn.Linear(in_features=config.max_seq_len * config.model_dim, 
                             out_features=config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

# txm djh 
# zhejiekewomen 结合代码讲解Transformer的模型结构
# 我们继续来看3.1节，编码器和解码器堆栈，这里说
# 编码器： 编码器由N=6个完全相同的层堆叠而成。
# 每一层都有两个子层。第一个子层是一个multi-head self-attention机制，第二个子层是一个简单的、位置完全连接的前馈网络。 
# 我们对每个子层再采用一个残差连接[11] ，接着进行层标准化[1]。
# 也就是说，每个子层的输出是LayerNorm(x + Sublayer(x))，其中Sublayer(x) 是由子层本身实现的函数。
# 为了方便这些残差连接，模型中的所有子层以及嵌入层产生的输出维度都为dmodel=512。
# 这段话一共提到了几个概念，首先是编码器的层级N=6,
# 图中就是左右两侧的N乘号,也就是说，图中左侧整体是一个Encoder或者编码器
# 右侧整体是一个解码器，关于Transformer的图形结构大家如果不理解可以结合之前我们的课件来看
# 代码中，也就是model.py文件中的Model类中的self.encoder和self.encoders两个实例化对象
# 其中self.encoders使用了nn.ModuleList方法，复制了num_encoders次，
# 也就是论文中的6个Encoder，代码中，可以根据需要调节Encoder的个数
# 我们继续来看图中的架构，首先是输入Inputs，
# 也就是(x1, x2,...xn)，经过InputEmbedding之后，
# 使用了一个加号拼接了PositionalEncoding
# 这个这里的Embedding我们讲过很多次了，
# 大家理解为统计所有训练、验证数据中的单词，形成一个唯一字典，
# 每个字典顺序有个编号，然后放回句子中，也就是说把句子中的每个单词转换为对应编号
# 句子就变成了很多编号的序列，代码中也就是类Model中的self.embedding 
# 这里有个if条件，就是说我们也可以使用一些已经训练好的词向量模型来做Embedding
# 简单理解就是换了另外一种Embedding的方式
# 回到论文，图中的positional Encoding，在代码中就对应了self.postion_embedding
# 然后论文中提到的，每一层都有两个子层
# 第一个子层是一个multi-head self-attention机制，第二个子层是一个简单的、位置完全连接的前馈网络。
# 对应代码中的Encoder类，我们知道self.encoder指的是一个encoder，
# 现在我们进到Encoder类中，可以看到，第一个子层是Multi_Head_Attention
# 第二个是Position_wise_Feed_Forward，与论文中提到的是一致的
# 然后论文中又提到，我们对每个子层再采用一个残差连接[11] ，接着进行层标准化[1]。
# 我们进到类Multi_Head_Attention中，在最后这里，out = out+x表明使用了残差连接
# out = self.layer_norm(out)表明使用了层标准化
# 同时我们也进到类Position_wise_Feed_Forward中，可以看到最后两行也是一样的
# 回到论文，后面这句话就是对前面的解释了，
# 表明我们先做了残差连接加了x，再进行层标准化也就是LayerNorm
# 最后是说，所有子层和嵌入层产生的输出维度都为dmodel=512
# 代码中，也就是两个类中初始化时传入的参数model_dim
# 回到论文中，解码层这里，
# 解码器同样由N = 6 个完全相同的层堆叠而成。 
# 除了每个编码器层中的两个子层之外，解码器还插入第三个子层，
# 该层对编码器堆栈的输出执行multi-head attention。
# 与编码器类似，我们在每个子层再采用残差连接，然后进行层标准化。
# 我们还修改解码器堆栈中的self-attention子层，以防止位置关注到后面的位置。 
# 这种掩码结合将输出嵌入偏移一个位置，确保对位置的预测 i 只能依赖小于i 的已知输出。
# OK,我们对比编码层来看，首先，解码器的个数6和编码层编码器的个数是一样多的，
# 虽然6这个数并没有特别实际的意义，
# 在每个编码器的层级中，子层的个数由两个变成了3个
# 在代码中，我们来看一下Encoder和Decoder的部分
# 我们来看一下，model_decoder.py文件与model.py文件不同之处在于，
# model.py只使用了Transformer的Encoder，模型用于文本分类，回归问题
# 也就是一条数据是由一条文本加一个标签组成
# model_decoder.py则加入了Decoder，模型适用于文本匹配、翻译等句子对的相关问题
# 假设我们文本翻译的数据是x, y，也就是说输入的句子是x，需要翻译为y
# 在这个文件里，我们来看一下Decoder和Encoder的不同，在forward函数里，
# 代码中类EncoderLayer的forward只有两行，
# 分别调用了Multi_Head_Attention和Position_wise_Feed_Forward方法
# 而在DecoderLayer则有三行，调用了Multi_Head_Attention两次，
# 调用Position_wise_Feed_Forward一次。
# 回到论文，最后一句说使用了掩码方式，这里我们的代码中没有这部分内容，
# 感兴趣的同学可以在我的github首页中，
# 在另一个项目Transformer-End-To-End中进行学习
# 论文中，我们来看3.2节，Attention
# Attention函数可以描述为将query和一组key-value对映射到输出，
# 其中query、key、value和输出都是向量。 
# 输出为value的加权和，其中分配给每个value的权重通过query与相应key的兼容函数来计算。
# 这里就是我们课件中所讲过的，也就是下面这个公式，缩放版的点积Attention
# 这部分的内容我们在课件中也都讲过了，
# 这部分内容中，注意第三段，其中说，加法attention使用具有单个隐藏层的前馈网络计算兼容性函数。
# 虽然两者在理论上的复杂性相似，但在实践中点积attention的速度更快、更节省空间，
# 因为它可以使用高度优化的矩阵乘法代码来实现。
# 这里作者应该是有所研究的，我们继续往后看，
# 下一段讨论了根号下dk的取值大小，作者也说怀疑这个数的大小会影响梯度
# 继续看一下3.2.2节，多头注意力机制
# 这里图与我们课件中的有所不同，第一张图是缩放版的点击Attention
# 我们来看一下，从输入x构造的三个矩阵Q,K,V，我们也对着论文中的公式(1)
# Q和K经过matmul，也就是点乘，然后经过scale，也就是除以根号下dk
# 经过mask和softmax，mask不是函数，而是一个判断条件，
# softmax也就是公式中的softmax函数，然后得到结果再和v进行点乘得到公式1最后的值
# 在我们的课件中，也就是其中的一个z1，
# 在多头注意力这种图中，Q,K,V分别先经过h个Linear层，构造成h个缩放版的点积Attention向量
# 最后使用拼接，也就是concat成一个向量传出去
# 代码中也就体现在多头注意力函数中，细节方面我们会在运行代码时再进行讲解
# 论文中下面这行说的，我们发现将query、key和value分别用不同的、
# 学到的线性映射复制h次到dk、dk和dv维效果更好，
# 而不是用d model维的query、key和value执行单个attention函数。
# 这句话涉及到张量的维度展开，大家先有个印象就可以了
# 然后后面说的也是拼接的工作，然后我们看后面这一段
# Multi-head attention允许模型的不同表示子空间联合关注不同位置的信息。
# 如果只有一个attention head，它的平均值会削弱这个信息。
# 这里的意思也就是课件中我们提到的，多头注意力机制通过多个层级捕捉句子中单词的相互之间的关系
# 这里的子空间也可以简单理解为多个层级
# 最后这段也就是课件中提到的z0,z1,z2,...z7
# 用了8个注意力头，设定的dk的大小是64，当然这里这些数字在代码中是可以随意调节的
# 论文中也只是设定了这些数值，得到了在某些数据集上比较好的效果
# 我们继续来看3.23节，
# 


class Encoder(nn.Module):
    """定义Encoder层"""
    def __init__(self, model_dim, num_heads, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(model_dim, num_heads, dropout)
        self.feed_forward = Position_wise_Feed_Forward(model_dim, hidden_size, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    """定义位置编码层"""
    def __init__(self, embedding_dim, max_seq_len, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        # 根据论文中的公式，构造PE矩阵
        self.position_encoding = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embedding_dim)) \
                                                for i in range(embedding_dim)] for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.position_encoding, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out

# 我们继续来看3.23节，
# 我们来重读一遍，在“编码器—解码器attention”层，query来自上面的解码器层，
# key和value来自编码器的输出。这允许解码器中的每个位置能关注到输入序列中的所有位置。
# 这模仿序列到序列模型中典型的编码器—解码器的attention机制，例如[38, 2, 9]。
# 这里这句话的意思是，在注意力机制方面，Transformer也延续了Encoder-Decoder的原理
# 也就是说，使用注意力机制，联系了输入和输出，使得输出的每个单词能够分别按不同程度关注到输入的每个单词
# 第二条是说，Transformer构建了self-attention，也就是自注意力机制，
# 这与传统的为单个序列建立RNN模型有所区别，最后一句编码器中的每个位置都可以关注编码器上一层的所有位置。
# 意思是，Transformer的多层级并没有损失每一层中自注意力的机制
# 即使是多层连接，每层和每层之间也具有注意力机制的连接
# 最后一条说，解码器中的self-attention层允许解码器中的每个位置都关注解码器中直到并包括该位置的所有位置。
# 这里翻译的有些令人费解，简单理解就是，解码层也有多层级自注意力机制
# 最后这句，我们需要防止解码器中的向左信息流来保持自回归属性。
# 我们在缩放版的点积attention中实现了，通过屏蔽softmax的输入中所有不合法连接的值（设置为-∞）
# 这里的意思是使用了一种mask机制，这一点我也不太理解，大家可以自己思考一下。。
# 然后我们继续看一下3.3节，基于位置的前馈网络
# 首先我们来看公式，说除了Attention的子层，
# 编码器和解码器中的每个层都包含一个完全连接的前馈网络，
# 该前馈网络单独且相同地应用于每个位置。这里这个位置，可以理解不同的层级
# 我们来看这个FFN函数的构造，它与普通的线性变换，y=wx+b不同之处在于
# 现在y=max(0, w1*x+b1)*w2+b2,意思就是说，我先做一个普通的线性变换y1=w1*x+b1
# 然后比较出来的结果y1和0的大小，取大的那个，那也就是说，如果y1大于0，就取y1，如果y1小于0，就取0
# 就是我们熟悉的relu函数呗。然后对做出的结果，再做一次线性变换
# 所以这里的公式也可以简化为，y=relu(w1*x+b1)*w2+b2,
# 或者，先线性变换，再取relu，再进行线性变换
# 或者从神经网络的角度来理解就是，先来一个全连接层，对第一层结果进行relu激活，然后再接一个全连接层
# 代码方面，我们也可以看一下函数Position_wise_Feed_Forward
# 来看forward函数，输入一个x，先来一个全连接层fc1，
# 然后是一个relu激活，然后再接一个全连接层fc2，最后也做了一下dropout，残差连接和层标准化
# 回到论文，继续来看3.4节，Embeddings和softmax，课件中我们也讲过了
# 使用了word Embedding和position Embedding，这里首先说的是
# 使用学习到的嵌入，也就是word Embedding，将输入词符和输出词符转换为维度为dmodel的向量。
# 还使用普通的线性变换和softmax函数将解码器输出转换为预测的下一个词符的概率。
# 这里说的分别是模型开始的输入层和最后的输出层的工作，
# 一方面做了词嵌入，另一方面在最后用了全连接和softmax
# 最后模型也使用了共享权重的技巧，提高了模型运行的效率
# 然后我们来看3.5节，之前课件中也有提到过，由于不适用RNN或者CNN，
# 对于一句话的处理不是从左到右建立模型，所以说并没有语言模型所强调的句子顺序的特征
# 就像论文中说的，模型学不到序列的顺序信息或者相对、绝对位置的信息
# 怎么办呢，这里的意思就是在word Embedding的后面加上了一个位置编码的东西，
# 词嵌入，衡量了一个单词在此表中的位置信息，位置嵌入，则衡量了这个词在当前句子中的位置信息
# 将“位置编码”添加到编码器和解码器堆栈底部的输入嵌入中。
# 论文中也提到了可以使用多种方式的位置编码，一种是可以放在模型中作为变量学习来用，
# 另一种是可以设置一个固定的编码方式，类似于word2vec一样的静态向量
# 论文中使用了一种正弦和余弦函数，这里可能有同学不太理解，我们着重讲一下
# 首先我们体会一下绝对位置和相对位置，在课件中，我们来看这样两句话：
# I really like this movie because it doesn't have an overhead history.
# I don't like this movie because it has an overhead history. 
# 这两句话的意思是完全相反的，但like在两句话中的绝对位置是相同的
# 所以说，再来看第一句中的like和doesn't，第二句中的don't和like
# 我们得出的结论是，在自然语言处理中，就像两个like绝对位置编码是没什么用的，
# 就像not和like的相对位置，相对位置的编码才起到关键作用
# OK，我们来看Transformer论文中的公式，翻译这个页面不太清楚，我们来看手写课件中
# sin函数中分子的pos，指的是单词或者token在句子中的位置，
# 如果句子长度为L，那么pos=0,1,2,...,L-1，比如在刚刚的例子中，I的pos=1，like的pos=2
# 然后分母指数中的i，指的是我们模型词向量的维度，比如论文中设定的词向量维度是512
# i就等于0,1,2,...,255, 2i可以取到0,2,4,...,512
# 这里我们就不讲特别清楚为什么会用这个公式，只是给一些结论
# 首先，当前位置t的位置表示PE(t, 2i)和其后k个单词后位置的位置表示PE(t+k, 2i)
# 它们的内积PE(t, 2i)*PE(t+k, 2i)是一个只与k相关的函数，与pos和i都无关
# 也就是说，可以使用这样的一个位置表示函数来衡量两个单词之间的相对位置
# 但是Transformer有个缺陷，就是在于使用这样的形式表示的相对位置的关系
# 还是没有先后关系，什么意思呢，就是当前x与x+k的关系和x+k与x的关系是一样的
# 或者说内积PE(t, 2i)*PE(t+k, 2i)和内积PE(t+k, 2i)*PE(t, 2i)是一样的
# 所以说，这里捕捉的是一种距离的表示，只是比绝对位置硬编码12345更好一些，而不是特别好的相对位置
# 因此后面BERT等模型也没有采用这种静态的position Embedding，
# 而是使用了动态可学习的位置嵌入，但是关于静态不需要学习的位置嵌入，
# 在其他论文中也有更好的工作，感兴趣的同学可以看看知乎的这篇文章
# 浅谈 Transformer-based 模型中的位置表示：https://zhuanlan.zhihu.com/p/92017824
# 回到论文中，我们来看第四节，为什么选择self-Attention
# 论文从三个角度比较了self-attention和循环神经网络、卷积神经网络的关系
# 第一个是模型复杂度，第二个是模型的并行性，第三个是模型捕捉长句子前后单词关系的能力
# 表1显示了对比的结果，这些结果都是从对代码的分析得来的
# 首先n是输入序列的长度，由于self-attention直接使用了两个句子向量之间的点积，
# 因此顺序操作的时间复杂度是O(1),RNN或者循环模型对于句子中的t个单词
# 是从h1>h2>...>ht进行构建的，因此顺序操作的时间复杂度是O(n)
# 经过代码的层级统计，层级的复杂度中self-attention是O(n^2*d)
# RNN是O(n*d^2),因此论文中也说了，
# 当序列长度n小于表示的维度d 时，self-attention比RNN快
# 反之，如果n大于d时，比如长文本的分类任务，RNN会更快一些
# 对于卷积神经网络来说，与其他二者的区别取决于选取的卷积核k的大小
# 因此，对于一些短文本，在同样的表示维度下，Transformer的性能也是很好的
# 继续来看第五节，训练的一些参数和方法
# 训练数据方面，使用了翻译数据集WMT-2014
# 硬件方面，使用了8个NVIDIA P100的GPU，
# 给出了训练一次的时间和总训练的时间，方便与其他模型进行对比
# 然后是优化器的选择，使用了Adam优化器，给了一些最终的参数取值
# 训练过程中也使用了动态变化的学习率，在训练的预热过程中线性增加学习率，随后按比例减小
# 正则化方面，或者为了防止过拟合，论文中使用了残差丢弃，课件中我们也说过，这是两个操作
# 首先是残差连接，然后是dropout，论文中也提到了是放在每个子层中，
# 在做层标准化之前做的这些残差连接和dropout
# 代码中，我们也可以看到在多头注意机制函数
# 和Position_wise_Feed_Forward函数的最后都是用了dropout和残差连接
# 论文中还提到了一种label smoothing，不太理解它的意思，
# 因此我们的代码中也没有使用这种方法防止过拟合
# 来看一下第六节的结果，模型对比了一些其他同一时期相关模型在同样翻译数据集上的结果
# 然后预测的效果提升了，Big版本的Transformer模型在翻译任务的BLEU指标
# 比之前最佳的模型的效果提高了2.0左右，从26左右提高到了28
# 训练成本方面，base模型把FLOPs质保降低了10倍，big模型也基本保持了最低
# 具体细节大家可以看一下论文中的一些其他参数
# 然后是模型变体这一小节，表三中给出了一些参数取值不同时，
# 最后的预测指标ppl、BLEU和参数量params也会因此而不同
# 比如参数有编码器的个数N，Embedding的维度d_model, 内部层的维度dff
# 注意力头的个数h

class Scaled_Dot_Product_Attention(nn.Module):
    '''定义缩放版的点积注意力层'''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    """定义多头注意力机制层"""
    def __init__(self, model_dim, num_heads, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        assert model_dim % num_heads == 0
        self.dim_head = model_dim // self.num_heads
        self.fc_Q = nn.Linear(model_dim, num_heads * self.dim_head)
        self.fc_K = nn.Linear(model_dim, num_heads * self.dim_head)
        self.fc_V = nn.Linear(model_dim, num_heads * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_heads * self.dim_head, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_heads, -1, self.dim_head)
        K = K.view(batch_size * self.num_heads, -1, self.dim_head)
        V = V.view(batch_size * self.num_heads, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_heads, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_heads)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    """"""
    def __init__(self, model_dim, hidden_size, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        # out = self.layer_norm(self.dropout(self.fc2(F.relu(self.fc1(x)))) + x)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out










