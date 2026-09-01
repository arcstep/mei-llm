# runtime/

正式推理系统发布面。`CURRENT.runtime` 仅在与正式模型权重组成可部署 release 后赋值；代码成果本身已经在此。

- [`mei-1.0-51m-route-v1/`](mei-1.0-51m-route-v1/) — Route-ID legacy，`legacy_frozen`
- [`mei-1.0-51m-needle2-v2/`](mei-1.0-51m-needle2-v2/) — Needle2 runtime，`implemented_in_code`，`not_a_toolcall_model_release`
- [`_shared/`](_shared/) — schema_render / normalizers / decode（两版 runtime 共用，唯一真源）
