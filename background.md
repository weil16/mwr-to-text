# 项目背景信息
乳腺癌是女性最常见的癌症之一。早期检测对改善治疗方式和降低死亡率至关重要。医学微波辐射测量（MWR）已成为癌症检测的替代及补充方法。MWR 通过 4 GHz 频率测量几厘米深度的内部组织温度。其检测癌症的原理是：癌细胞快速生长需要更多能量，因此比正常细胞产生更多热损耗。机器学习（尤其是深度神经网络）的集成在提升 MWR 诊断精度方面展现出巨大潜力。然而，医学应用中深度神经网络的 “黑箱” 特性带来了挑战。临床医生和患者都希望理解神经网络预测的推理过程，尤其是在影响生命的医疗决策中。“MWR 到文本”的概念应运而生：这是一种新颖的方法，用于弥合 MWR 数据的复杂视觉表征与人类可理解文本描述之间的鸿沟。MWR 乳腺数据包含皮肤表面和几厘米深度内部组织的温度读数。每个腺体有 9 个测量点，腋窝区域有 2 个，胸腔下方有 2 个参考点，每个病例共有 44 个温度读数。患者根据温度不对称性按 0 到 5 的等级评分。该数据集包含超过 28,000 个标记病例。

# 主要功能
基于不同患者乳腺温度测量数据给出患癌风险等级判断（0，1）的同时生成个性化医疗诊断结论解释判断的依据（适配 MWR 到文本模型，实现联合文本与分类预测）

# 模型结构
 1. 模型是一个多任务模型，要求输入一串标签加数值构成的文本，输出对cancer_risk的预测（二分类，0或1）和一段医疗诊断说明（文本生成），通过微调T5-small模型实现

# 模型细节
 1. 要求实现自定义的dataset和dataloader
 2. 分类任务使用交叉熵函数，样本的监督标签从data_th_scale.csv中的cancer_risk列读取
 3. 文本生成任务使用T5自带的损失函数，标签为data_th_scale.csv中的Conclusion (Tr)列读取
 4. 模型总的loss为分类任务的loss和文本生成任务的loss加权求和得到的
 5. 要求使用lora，混合精度训练，梯度累积之类的加速训练的方法
 6. 要求微调涉及的各种超参数都可调

# 数据集细节
 1. temperature reading是从data_th_scale.csv中读取的，包含以下的column："Hormonal medications", "Cancer family history", "Breast operations", "Conclusion (Tr)", "r:AgeInYears", "Cycle", "Day from the first day", "Num of pregnancies", "T1 int", "T2 int", "T1 sk", "T2 sk", "L0 int", "L1 int", "L2 int", "L3 int", "L4 int", "L5 int", "L6 int", "L7 int", "L8 int", "L9 int", "R0 int", "R1 int", "R2 int", "R3 int", "R4 int", "R5 int", "R6 int", "R7 int", "R8 int", "R9 int", "L0 sk", "L1 sk", "L2 sk", "L3 sk", "L4 sk", "L5 sk", "L6 sk", "L7 sk", "L8 sk", "L9 sk", "R0 sk", "R1 sk", "R2 sk", "R3 sk", "R4 sk", "R5 sk", "R6 sk", "R7 sk", "R8 sk", "R9 sk" 
 2. 数据集列名解释：R0 int - R9 int and L0 int - L9 int: The internal temperature readings of the right and left breast respectively in Celsius
    R0 sk - R9 sk and L0 sk - L9 sk: The skin temperature readings of the right and left breast respectively
    T1 and T2 are reference measurements under the chest
    Cycle: The typical duration of the patient's menstrual cycle. If the value is not available (n/a), -1 or missing, it may indicate that the patient is likely in menopause or pregnant, depending on their age. If a range of values is provided, it suggests the patient has a very irregular cycle
    Day of Cycle: The days from the first day of a period, where applicable
 3. 数据集划分为train(70%), val(15%), test(15%)

# 数据预处理
 1. 基于data_th_scale.csv添加一列，列名为 “cancer_risk”，对r:Th列中所有等于0的样本，其对应的cancer_risk也为0，对r:Th列中数值不为0的样本，其对应的cancer_risk为1
 2. 对R0 int - R9 int and L0 int - L9 int的温度数据基于T1 int和T2 int做归一化
 3. 对R0 sk - R9 sk and L0 sk - L9 sk列的温度数据基于T1 sk和T2 sk做归一化
 4. 归一化的方式如代码所示：
 def noramalise(df: pd.DataFrame, label_tag: str, ref_label: str) -> pd.DataFrame:
        def line_function(x, A, B):
            return A * x + B

        def transform(temperature, A, refAvg, ref):
            return temperature + A * (refAvg - ref)
        
        ref_mean = df[ref_label].mean(axis=0)
        
        for i in range(10):
            label = f"{label_tag}{i} int"
            if label in df.columns:
                A, B = curve_fit(line_function, df[ref_label].values, df[label].values)[0]
                df[label] = np.vectorize(transform)(df[label], A, ref_mean, df[ref_label])
        return df

# 评估标准
 1. 分类任务评估标准为accuracy，f1，sensitivity，specificity，auc
 2. 文本生成任务评估标准为bert_f1_score，meteor_mean_score