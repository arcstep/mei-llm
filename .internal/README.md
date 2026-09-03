# Internal support

这里不是业务域，也不允许保存唯一实现：

- `registry/`：由五域事实生成的机器索引、schema 和旧路径迁移映射；
- `experiments/`：Qwen 等不属于当前主产品的实验说明。

训练、评估、编排和发布源码已提升到可见的 `model-factory/`；CLI/registry 门面在 `src/`；
跨域测试归各自工厂。若在本目录发现新的 `.py` 活跃算法文件，布局门禁应直接失败。
