# LLM Stack Blocks Entry

Use this path when the stack order must be parsed by the VLM/LLM from a natural
language instruction.

```bash
cd /home/wxm/code/RobotStackDemo

python3 tools/workflows/stack_demo_pipeline.py \
  --force-llm-decision \
  --instruction "以红色积木为底，把绿色积木放到红色上面，再把蓝色积木放到绿色上面" \
  --output-dir runtime/llm_stack_blocks_test \
  --model qwen2.5vl:7b-q4_K_M \
  --tcp-offset-tool -0.015 0 0.15
```

Add `--execute --yes` only when the robot, gripper, RealSense, TF, MoveIt, and
Ollama model are ready.

Generated decision files:

- `initial_order_llm/llm_input.json`
- `initial_order_llm/llm_scene_graph_decision_raw.json`
- `initial_order_llm/llm_scene_graph_decision.json`
- `stack_blocks_decision.json`

The `--force-llm-decision` flag forces the VLM/LLM call path for the initial
symbolic stack order. Before execution, object ids are still repaired and
validated against detector hard priors using explicit color words in the
instruction. This follows the safer Scene-Graph-Benchmark behavior: the model
may describe the order, but it is not trusted to invent or renumber detector
ids.

The closed-loop pick/place, second-snapshot XY correction, place verification,
and safety checks remain unchanged.
