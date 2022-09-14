# BSM1_cn

将Modelica模型(BSM1)转换为Python OpenAI Gym环境，并使用人工智能的强化学习代理优化操作成本。

本README提供了关于如何:

1. 使用[Open Modelica](https://www.openmodelica.org/)从Python ([OMPython](https://github.com/OpenModelica/OMPython))运行Modelica模型

2. 将模型转换为[FMU](https://fmi-standard.org/)格式，并使用[PyFMI](https://github.com/modelon-community/PyFMI)在Python中直接模拟它。

3. 为FMU创建一个OpenAI环境，并使用强化学习代理与模型交互

这里的代码是用来模拟[污水处理算法ASM1/ASM2/ASM3](https://www.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf) (BSM1)的Modelica实现的，可以在这个资源库中的[./WasteWaterResearch/BSM1/](./ wasteresearch /BSM1/)目录中找到它，它改编自[Félix Hernández del Olmo](https://github.com/felixhdelolmo/WasteWaterResearch)的工作。此外，它的编译FMU已经在这个存储库中可用([BSM1.BSM1.fmu](./BSM1.BSM1.fmu))，它可以直接在Windows 64位系统下使用(它是在这个存储库中编译的)，而不需要重新编译它(在任何情况下，生成FMU的指令已经包含在本README中)。最后，即使用户对BSM1不感兴趣，该存储库也提供了工具和说明，以适应任何Modelica模型，将其集成到OpenAI Gym API中，并用于在其上训练强化学习代理。

## 0.制备步骤
 - 下载并安装适用于 Python 3.7 的 Miniconda：https ://docs.conda.io/en/latest/miniconda.html
 - 安装后，打开 Anaconda 提示符，并创建 Modelica 环境：conda create -n modelica python=3.7
 - 激活它：conda activate modelica
 - 安装一些需要的库：
```{bash}
conda install numpy matplotlib psutil pyzmq scipy future git pandas seaborn scikit-learn tqdm
conda install -c conda-forge ffmpeg
```
 - 将此存储库克隆到您的计算机并导航到它：:
```{bash}
git clone https://github.com/OscarPellicer/BSM1_gym.git
cd BSM1_gym
```

## 1.通过OMPython与OpenModelica模型交互
下载并安装 OpenModelica：https ://www.openmodelica.org/
要从 Python（非交互式）运行 Modelica 模型，我们可以安装和使用 OMPython 库：
安装：python -m pip install -U https://github.com/OpenModelica/OMPython/archive/master.zip
要查看以下部分中的代码生成的任何绘图，建议在 Jupyter Notebook / Jupyter 实验室中运行代码。
要安装 Jupyter 实验室：conda install jupyter jupyterlab.
然后，jupyter notebook在控制台中运行，并导航到http://localhost:8888/lab。然后单击文件 > 新建 > 笔记本，开始编码！
我们为 Python 中的 OpenModelica 命令创建了一个简化的运行程序Runner，可以在utils.py. 弹跳球模型测试：
```{bash}
from utils import Runner
R= Runner()
R.run("loadModel(Modelica)")
R.run("loadFile(getInstallationDirectoryPath() + \"/share/doc/omc/testmodels/BouncingBall.mo\")")
R.run("instantiateModel(BouncingBall)")
R.run("simulate(BouncingBall, stopTime=3.0)")
```
这将生成一个.mat可以使用DyMat读取的输出文件，该文件已包含在此存储库中。此外，类MatHandler使用 DyMat 来简化读取和绘制模拟结果的过程：
```{bash}
from utils import MatHandler
m= MatHandler('BouncingBall_res.mat')
print('Variable names:', m.names)
m.plot(['h', 'v', 'der(v)'], plot_indep=False)
```
## 2. 创建 FMU 并使用 PyFMI 对其进行仿真
PyFMI 允许在 Python 中模拟 FMU
安装pyfmi：（conda install -c conda-forge pyfmi注意：如果运行 jupyter lab，你现在需要打开一个新的 Anaconda 终端并激活modelica环境，或者关闭 Jupyter）
然后，使用 OMPython 从 Python 创建 FMU：
```{bash}
model_name= "BouncingBall"
R.run("loadModel(Modelica)")
R.run("loadFile(getInstallationDirectoryPath() + \"/share/doc/omc/testmodels/BouncingBall.mo\")")
R.run("instantiateModel(BouncingBall)")
R.run('translateModelFMU(%s, version="2.0", fmuType = "me")'%model_name)
并使用 PyFMI 模拟它：
from pyfmi import load_fmu
model= load_fmu('%s.fmu'%model_name)
res= model.simulate(final_time=3)
```
我们可以为我们的自定义模型做同样的事情
```{bash}
# Create a FMU using OMPython
base_path= 'WasteWaterResearch'
model_name= 'BSM1.BSM1'
R.run('loadModel(Modelica,{"3.2.3"},true,"",false)')
R.run('loadFile("%s/WasteWater/package.mo","UTF-8",true,true,false)'%base_path)
R.run('loadFile("%s/BSM1/package.mo","UTF-8",true,true,false)'%base_path)
R.run('disableNewInstantiation()')
R.run('translateModelFMU(%s, version="2.0", fmuType = "me")'%model_name)

# Import PyFMI, set the options so that sparse solver is not used, and simulate
from pyfmi import load_fmu
model = load_fmu('%s.fmu'%model_name)
opts = model.simulate_options() 
opts["CVode_options"]["linear_solver"] = "DENSE"
res = model.simulate(final_time=1, options=opts)
pyFMI允许一个简单的结果界面，而无需读取生成的.mat文件：
# To see the name of all variables
res.keys()
# To access the values of one of them
res['time']
# To access final value
res.final('time')
# To get the description
res.result_data.description[res.keys().index('time')]
```
## 3. 创建一个OpenAI健身房环境，并在其上培训一个增强学习代理

安装OpenAI健身房:' conda Install -c conda-forge Gym '

建立一个环境(BSM1)，并培训一个代理以优化其运行成本的例子可以在[training .ipynb](training .ipynb)中找到。

-为了使用深度学习代理，你也需要安装pytorch。它也需要运行[Training.ipynb](Training.ipynb)。您可以在这里执行说明:https://pytorch.org/get-started/locally/

-显然，要运行Notebook，必须首先安装Jupyter Notebook(参见[上面](# interaction -with-openmodelica-models-through-ompython))。

-请注意，BSM1模拟有时会失败，因为代理进行了一系列的操作，导致模型中被除零。这是一种罕见的情况，不幸的是还没有解决。如果出现这种情况，请重新运行模拟。

-如果你想让这个库适应你自己的需要，你应该首先实现一个类似于' BSM1Envs/bsm1_env.py '中的' BSM1Env '类。这个类实现了“ModelicaEnv”类，而这个类又继承自“gym”。Env '，它是所有Gym环境的父类。

-你还应该修改[Training.ipynb](Training.ipynb)中的一些变量，例如' base_path '， ' model_name '， ' action_name '， ' env_name '， ' entry_point '， ' output_names '，使它们适应你自己的模型。

-存储库中其余的Python文件由[Training.ipynb](Training.ipynb)使用，并提供以下功能:

- ' agents.py ':该文件包含一些最常见的q类代理的实现，使用

为深层特工准备的火药。

- ' wrapper .py ':它包含一些用于OpenAI环境的有用包装器，以帮助集成它们

用不同种类的药剂。

- ' training.py ':这个文件包含让代理交互和从环境中学习的所有代码

也可以画出变量值随时间的变化。

这项工作是我在[联合国开发大学]计算机科学学位(https://www.uned.es/)的学位结业项目的一部分，可以在[这个存储库](https://github.com/OscarPellicer/BSM1_gym/blob/main/TFG%20Oscar%20Jos%C3%A9%20Pellicer%20Valero.pdf)(西班牙语)中找到。
