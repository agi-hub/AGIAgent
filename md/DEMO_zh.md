# AGI Agent 演示案例

本文档展示了 AGI Agent 在各种任务场景中的能力。每个案例都包含具体的任务描述和生成的输出文件。

所有示例可在以下链接找到： 
https://drive.google.com/drive/folders/14SwKNCCUYWODzxjb-DdCzeHo0VdYTSt9?usp=sharing

## 英文示例

| Case Name | Description | Output Files |
|-----------|-------------|--------------|
| `intro_flash_atten` | provide an introduction to flash attention in word doc. |  <img src="images/flash_atten_report.png" alt="flash_atten_report.png" width="200"> |
| `deepseek_chat_svd` | design an SVD experiment, collect the results, and write a report | <img src="images/svd_results.png" alt="svd_results.png" width="200"><br/><br/><br/>📁 requirements.txt<br/>📋 report.md<br/>📁 report.html |
| `equibrium_prop_latex_pdf` | write an introduction of equibrium propagation in latex and compile it to pdf | <img src="images/eq_pdf.png" alt="eq_pdf.png" width="200"> |
| `yolo_setup` | Get the yolov8 repo. and provide a demo| <img src="images/yolodemo.png" alt="yolodemo.png" width="200"> |
| `sorting_in_10_languages` | Write sorting algorithm in 10 languages | source code files|
| `songs` | write a song for programmers | 📁 programmer_song_lyrics.txt<br/>📋 programmer_song.md|

## 相同需求由不同模型执行的结果对比
| Case Name | Description | Output Files |
|-----------|-------------|--------------|
| `deepseek_chat_scikit_demo` | install scikit and write a demo program, saved the results including figures and illustrations in word | 📝 scikit_demo_results.docx |
| `doubao_1_5_pro_scikit_demo` | the same | <img src="images/iris_scatter_plot.png" alt="iris_scatter_plot.png" width="200"><br/><br/><br/>📝 scikit_demo_results.docx |
| `claude_4_sonnet_scikit_demo` | the same| <img src="images/clustering_results.png" alt="clustering_results.png" width="200"><br/><br/>+3 more images<br/><br/><br/>📝 sklearn_demo_report.docx |
| `qwen7b_scikit_demo` | the same | 📋 scikit_demo.md<br/><br/>+20 more documents |

## 中文示例

| Case Name | Description | Output Files |
|-----------|-------------|--------------|
| `claude_4_sonnet_pacman_game_3_loops` | 写一个html版本的pacman | 📁 index.html |
| `model_analysis_zh` | 根据transformer仓库,分析qwen2.5-omni的网络结构,包括 主要架构、核心参数、各模块结构及核心算子 | 📋 06_comparative_analysis.md<br/>📋 08_future_outlook.md<br/><br/>+51 more documents |
| `news_beans_price_zh` | 最近的大豆价格有什么变化趋势?调研并写个word总结 | 📋 最近大豆价格变化趋势报告.md<br/>📝 最近大豆价格变化趋势报告.docx<br/><br/>+177 more documents |
