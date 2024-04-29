## 复现说明


### 1. 从huggingface下载模型
参考网络教程（[例](https://zhuanlan.zhihu.com/p/685765714)）更换huggingface镜像，下载[SDXL模型](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)；

### 2. 修改下载后的模型配置

模型默认下载路径在`~/.cache/huggingface/hub/models--MODEL-NAME/snapshots/HASH`，找到子目录下`unet/config.json`修改模型大小。

参考：
```diff
"transformer_layers_per_block": [
-    1,
-    2,
-    10
+    1,
+    1,
+    1
],
```

### 3. 修改后运行`train.sh`脚本

`PYTHONPATH`、`LOG_DIR`等路径相关的变量修改为对应本地代码仓路径

**\[optional\]** `MODEL_NAME`可以修改为本地路径