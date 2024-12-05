
在`Streamlit`中，`Form`组件是一种特殊的UI元素，允许用户输入数据而不立即触发应用的重新运行。


这对于创建需要用户输入多个参数后再进行处理的交互式表单非常有用。


# 1\. 概要


`Form`组件的主要作用是在一个表单内集中处理多个用户输入，使得数据收集和验证更加高效和直观。


通过`Form`组件，开发者可以创建包含多个输入控件（如文本输入框、下拉选择框等）的表单，用户可以在表单内一次性填写所有必要的信息，然后提交。


这避免了传统表单提交时每次输入都会触发页面刷新的问题，从而提高了用户体验和应用的交互性。


根据`Form`组件的特点，在类似下面这些场景中，我们可以考虑使用`Form`：


1. **用户注册与登录**：通过`Form`组件构建一个包含用户名、密码、邮箱等多个输入组件，以及一个提交按钮的页面，并在用户点击提交按钮后才开始进行验证和处理。
2. **数据查询与筛选**：通过`Form`组件可以包含多个选择框、输入框等组件，用于收集用户的查询或筛选条件。
3. **参数配置与设置**：在构建复杂的Web应用程序时，可能需要用户配置或设置一些参数，这些参数可能包括算法参数、界面样式等。通过`Form`组件，可以集中展示和配置这些参数。
4. **多步骤表单处理**：通过`Form`组件，开发者可以创建包含多个步骤的表单，并在用户完成每个步骤后收集相应的数据。
5. **动态表单生成**：在某些高级应用场景中，可能需要根据用户的选择或输入动态生成表单。例如，在构建在线问卷时，可能需要根据用户的选择展示不同的问题。


总之，`Streamlit`的`Form`组件在很多应用场景中都发挥着重要作用，特别是在需要收集和处理多个用户输入的场景中表现尤为突出。


# 2\. 主要参数


`Form`组件的参数很简单，主要用来简单的控制样式和提交的行为。




| **名称** | **类型** | **说明** |
| --- | --- | --- |
| key | str | 组件名称，具有唯一性 |
| clear\_on\_submit | bool | 用户提交后，表单内的所有组件是否都重置为默认值 |
| enter\_to\_submit | bool | 当用户在与表单内的组件交互时，按`Enter`键时是否提交表单 |
| border | bool | 是否在窗体周围显示边框 |


`Form`组件本身并不直接接受各种组件来作为参数，但表单内部可以包含多种输入组件，如文本框（`st.text_input`）、选择框（`st.selectbox`）、滑块（`st.slider`）等。


此外，`Form`组件需要配合`st.form_submit_button`来创建一个提交按钮。


# 3\. 使用示例


下面通过一些根据实际场景来简化的示例，演示`Form`组件的使用方式。


## 3\.1\. 数据预处理参数设置


在数据分析或机器学习项目中，数据预处理是一个关键步骤。


我们可以使用`Form`组件来让用户选择数据预处理的参数，如缺失值处理方法和特征缩放方法。



```
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 加载示例数据集
data = pd.DataFrame(
    {"feature1": [1, 2, None, 4, 5], "feature2": [10, 20, 30, None, 50]}
)


# 定义表单提交后的回调函数
def preprocess_data(fill_method, scale_method):
    if fill_method == "mean":
        data.fillna(data.mean(), inplace=True)
    elif fill_method == "median":
        data.fillna(data.median(), inplace=True)
    else:
        data.dropna(inplace=True)

    if scale_method == "standard":
        scaler = StandardScaler()
    elif scale_method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler:
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
        )
        st.write(data_scaled)
    else:
        st.write(data)


# 创建表单
with st.form(key="preprocess_form"):
    fill_method = st.selectbox(label="缺失值处理", options=["mean", "median", "drop"])
    scale_method = st.selectbox(
        label="特征缩放", options=["standard", "minmax", "none"]
    )
    submitted = st.form_submit_button(label="提交")
    if submitted:
        preprocess_data(fill_method, scale_method)

```

运行效果如下，【提交】按钮点击后才会刷新页面。


![](https://img2024.cnblogs.com/blog/83005/202412/83005-20241204120847072-228784474.gif)


## 3\.2\. 机器学习模型超参数调优


在训练机器学习模型时，超参数的选择对模型性能有很大影响。


我们可以使用`Form`组件来让用户选择模型的超参数，并展示模型在验证集上的性能。



```
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载示例数据集
data = load_iris()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 定义表单提交后的回调函数
def train_model(n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    st.write(f"准确率: {accuracy:.2f}")


# 创建表单
with st.form(key="model_form"):
    n_estimators = st.number_input(
        label="Estimators 数量",
        min_value=10,
        max_value=200,
        step=10,
        value=100,
    )
    max_depth = st.number_input(
        label="最大深度",
        min_value=1,
        max_value=20,
        step=1,
        value=10,
    )

    submitted = st.form_submit_button(label="开始训练")
    if submitted:
        train_model(n_estimators, max_depth)

```

运行界面如下，点击【开始训练】按钮后显示训练后模型的准确率。


![](https://img2024.cnblogs.com/blog/83005/202412/83005-20241204120847261-1572216292.png)


# 4\. 总结


总的来说，`Streamlit`的`Form`组件能够帮助我们简化表单的创建和数据收集的过程，使我们能够轻松构建具有复杂交互功能的数据应用。


 本博客参考[slower加速器](https://chundaotian.com)。转载请注明出处！
